"""Match boundary scanner for multi-match recordings.

Detects individual match boundaries in long recordings by reading
the in-game timer from extracted frames.
"""

import logging
import re
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.battle_analyzer import BattleAnalyzer, _half_resize
from src.cv_detector import read_timer
from src.frame_extractor import extract_frames
from src.highlight_detector import _isotonic_decreasing, _median_smooth

ScanProgressCallback = Callable[[int, int], None]  # (frames_done, frames_total)

logger = logging.getLogger(__name__)

TIMER_SCAN_SYSTEM_PROMPT = """\
あなたはスプラトゥーンのゲーム画面の上部から試合タイマーを読み取る専門AIです。
この画像はゲーム画面の上部中央（上15%・中央40%幅）をクロップしたものです。

■ 読み取り対象:
- 画面上部中央に表示されている試合の残り時間タイマー
- タイマーは「M:SS」形式（例: 4:52, 2:10, 0:30）で表示されている
- 試合中でない場合（メニュー画面、リザルト画面等）はタイマーが表示されていない

■ ルール:
- タイマーの残り時間の数値のみを読み取ること
- タイマーが見えない、または不明瞭な場合は null を返すこと
- タイマー以外の情報は一切不要

■ 出力フォーマット（JSONのみ、他のテキスト不可）:
{
  "timer_remaining": "M:SS" | null
}
"""

TIMER_SCAN_USER_PROMPT = "この画像（ゲーム画面の上部中央）から試合タイマーの残り時間を読み取ってJSON形式で回答してください。"

# Clustering tolerance in seconds for grouping frames into the same match
CLUSTER_TOLERANCE = 30.0


@dataclass
class MatchInfo:
    """Detected match boundary information."""

    start_seconds: float
    duration_seconds: int
    duration_type: str  # "5min" or "3min"


def parse_timer(timer_str: str) -> float | None:
    """Parse a timer string in M:SS format to seconds.

    Args:
        timer_str: Timer string like "4:52", "2:10", "0:30".

    Returns:
        Timer value in seconds, or None if parsing fails.
    """
    match = re.match(r"^(\d+):(\d{1,2})$", timer_str.strip())
    if not match:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    if seconds >= 60:
        return None
    return minutes * 60 + seconds


def determine_rule(timer_seconds: float) -> tuple[int, str]:
    """Determine match rule type from timer remaining value.

    Args:
        timer_seconds: Remaining time in seconds.

    Returns:
        Tuple of (total_duration_seconds, duration_type).
        5min rule if timer > 180s, 3min rule if timer <= 180s.
    """
    if timer_seconds > 180:
        return 300, "5min"
    return 180, "3min"


def calc_match_start(frame_timestamp: float, timer_seconds: float, total_duration: int) -> float:
    """Calculate estimated match start time.

    Args:
        frame_timestamp: Timestamp of the frame in the video (seconds).
        timer_seconds: Timer remaining value in seconds.
        total_duration: Total match duration in seconds (300 or 180).

    Returns:
        Estimated match start time in seconds.
    """
    elapsed = total_duration - timer_seconds
    return frame_timestamp - elapsed


@dataclass
class TimerReading:
    """A single timer reading from one frame."""

    frame_timestamp: float
    timer_seconds: float
    total_duration: int
    duration_type: str
    match_start: float


@dataclass
class ScanResult:
    """Result of a match scan: detected matches and raw readings."""

    matches: list[MatchInfo]
    readings: list[TimerReading]


def _expand_candidates(readings: list[TimerReading]) -> list[TimerReading]:
    """Expand each reading into candidates for both 5min and 3min rules.

    For readings where only one rule is possible (timer > 180s => must be 5min),
    only that candidate is produced.  For ambiguous readings (timer <= 180s),
    both 5min and 3min candidates are produced.  Clustering later picks the
    consistent interpretation.
    """
    candidates: list[TimerReading] = []
    for r in readings:
        if r.timer_seconds > 180:
            candidates.append(r)
        else:
            for total, dtype in [(300, "5min"), (180, "3min")]:
                start = calc_match_start(
                    r.frame_timestamp, r.timer_seconds, total
                )
                candidates.append(
                    TimerReading(
                        frame_timestamp=r.frame_timestamp,
                        timer_seconds=r.timer_seconds,
                        total_duration=total,
                        duration_type=dtype,
                        match_start=start,
                    )
                )
    return candidates


def cluster_readings(
    readings: list[TimerReading],
) -> tuple[list[MatchInfo], list[TimerReading]]:
    """Group timer readings into matches by clustering estimated start times.

    For each frame, both 5min and 3min interpretations are considered.
    Clustering selects the consistent interpretation automatically.

    Args:
        readings: List of timer readings.

    Returns:
        Tuple of (matches sorted by start_seconds, resolved readings).
    """
    if not readings:
        return [], []

    candidates = _expand_candidates(readings)
    sorted_candidates = sorted(candidates, key=lambda r: r.match_start)

    clusters: list[list[TimerReading]] = []
    current_cluster: list[TimerReading] = [sorted_candidates[0]]

    for reading in sorted_candidates[1:]:
        prev = current_cluster[-1]
        if abs(reading.match_start - prev.match_start) <= CLUSTER_TOLERANCE:
            current_cluster.append(reading)
        else:
            clusters.append(current_cluster)
            current_cluster = [reading]
    clusters.append(current_cluster)

    # Count unambiguous readings (timer > 180 => must be 5min) per cluster.
    unambiguous_count: dict[int, int] = {}
    for ci, cluster in enumerate(clusters):
        unambiguous_count[ci] = sum(
            1 for r in cluster if r.timer_seconds > 180
        )

    def _cluster_priority(ci: int) -> tuple[int, int, int]:
        """Higher = better: (size, has_unambiguous, prefer_shorter)."""
        return (
            len(clusters[ci]),
            1 if unambiguous_count[ci] > 0 else 0,
            1 if clusters[ci][0].total_duration == 180 else 0,
        )

    # Deduplicate: keep only the cluster each frame_timestamp belongs to
    # with the best priority.
    frame_best: dict[float, tuple[int, TimerReading]] = {}
    for ci, cluster in enumerate(clusters):
        for r in cluster:
            prev = frame_best.get(r.frame_timestamp)
            if prev is None or _cluster_priority(ci) > _cluster_priority(prev[0]):
                frame_best[r.frame_timestamp] = (ci, r)

    # Rebuild clusters using only best assignments
    best_clusters: dict[int, list[TimerReading]] = {}
    for ci, r in frame_best.values():
        best_clusters.setdefault(ci, []).append(r)

    matches: list[MatchInfo] = []
    for cluster in best_clusters.values():
        if len(cluster) < 2:
            continue
        starts = [r.match_start for r in cluster]
        median_start = statistics.median(starts)

        type_counts: dict[str, int] = {}
        for r in cluster:
            type_counts[r.duration_type] = (
                type_counts.get(r.duration_type, 0) + 1
            )
        best_type = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
        best_duration = 300 if best_type == "5min" else 180

        matches.append(
            MatchInfo(
                start_seconds=max(0.0, round(median_start, 1)),
                duration_seconds=best_duration,
                duration_type=best_type,
            )
        )

    matches.sort(key=lambda m: m.start_seconds)
    resolved = sorted(
        [r for _, r in frame_best.values()],
        key=lambda r: r.frame_timestamp,
    )
    return matches, resolved


def _normalize_timer_readings(readings: list[TimerReading]) -> list[TimerReading]:
    """Apply median smoothing + isotonic decreasing regression to timer values.

    Groups readings by initial cluster assignment (sorted by match_start,
    grouped within CLUSTER_TOLERANCE), then normalizes timer_seconds within
    each group so they monotonically decrease over time.
    """
    if not readings:
        return readings

    # Sort by match_start to form initial groups
    sorted_readings = sorted(readings, key=lambda r: r.match_start)

    # Group within CLUSTER_TOLERANCE
    groups: list[list[TimerReading]] = []
    current_group: list[TimerReading] = [sorted_readings[0]]
    for r in sorted_readings[1:]:
        if abs(r.match_start - current_group[-1].match_start) <= CLUSTER_TOLERANCE:
            current_group.append(r)
        else:
            groups.append(current_group)
            current_group = [r]
    groups.append(current_group)

    normalized: list[TimerReading] = []
    for group in groups:
        # Sort by frame_timestamp within each group
        group.sort(key=lambda r: r.frame_timestamp)

        # Extract timer_seconds as int values for smoothing functions
        timer_values: list[int | None] = [int(r.timer_seconds) for r in group]

        # Apply median smoothing then isotonic decreasing regression
        smoothed = _median_smooth(timer_values)
        fitted = _isotonic_decreasing(smoothed)

        for i, r in enumerate(group):
            norm_timer = float(fitted[i]) if fitted[i] is not None else r.timer_seconds
            total_duration, duration_type = determine_rule(norm_timer)
            match_start = calc_match_start(r.frame_timestamp, norm_timer, total_duration)
            normalized.append(
                TimerReading(
                    frame_timestamp=r.frame_timestamp,
                    timer_seconds=norm_timer,
                    total_duration=total_duration,
                    duration_type=duration_type,
                    match_start=match_start,
                )
            )

    return normalized


class MatchScanner:
    """Scan a video recording to detect individual match boundaries."""

    def __init__(self, analyzer: BattleAnalyzer, interval: float = 20.0) -> None:
        """Initialize the scanner.

        Args:
            analyzer: BattleAnalyzer instance for frame analysis.
            interval: Frame extraction interval in seconds (default: 20).
        """
        self.analyzer = analyzer
        self.interval = interval

    def scan(
        self,
        video_path: str | Path,
        progress_callback: ScanProgressCallback | None = None,
    ) -> ScanResult:
        """Scan a video for match boundaries.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback (frames_done, frames_total).

        Returns:
            ScanResult with detected matches and raw timer readings.
        """
        video_path = Path(video_path)

        frames = extract_frames(
            video_path=video_path,
            interval_seconds=self.interval,
            no_save=True,
        )

        total_frames = len(frames)
        readings: list[TimerReading | None] = [None] * total_frames
        done_count = [0]

        def _analyze_one(index: int) -> None:
            timestamp_sec = index * self.interval
            ts_label = self._format_timestamp(timestamp_sec)

            frame = _half_resize(frames[index])
            h, w = frame.shape[:2]
            upper = frame[: int(h * 0.15), :, :]
            left = int(w * 0.3)
            right = int(w * 0.7)
            upper_half = upper[:, left:right, :]

            timer_str = read_timer(upper_half)
            timer_value = parse_timer(timer_str) if timer_str else None
            logger.debug("Timer scan at %s: %s -> %s", ts_label, timer_str, timer_value)

            if timer_value is not None:
                total_duration, duration_type = determine_rule(timer_value)
                match_start = calc_match_start(timestamp_sec, timer_value, total_duration)
                readings[index] = TimerReading(
                    frame_timestamp=timestamp_sec,
                    timer_seconds=timer_value,
                    total_duration=total_duration,
                    duration_type=duration_type,
                    match_start=match_start,
                )

            done_count[0] += 1
            if progress_callback:
                progress_callback(done_count[0], total_frames)

        with ThreadPoolExecutor(max_workers=self.analyzer.concurrency) as executor:
            futures = [executor.submit(_analyze_one, i) for i in range(total_frames)]
            for future in as_completed(futures):
                future.result()

        valid_readings = [r for r in readings if r is not None]
        logger.info(
            "Timer scan complete: %d/%d frames had valid timer readings",
            len(valid_readings),
            total_frames,
        )

        valid_readings = _normalize_timer_readings(valid_readings)

        matches, resolved_readings = cluster_readings(valid_readings)
        return ScanResult(matches=matches, readings=resolved_readings)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"
