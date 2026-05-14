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

from src.battle_analyzer import BattleAnalyzer
from src.frame_extractor import extract_frames

ScanProgressCallback = Callable[[int, int], None]  # (frames_done, frames_total)

logger = logging.getLogger(__name__)

TIMER_SCAN_SYSTEM_PROMPT = """\
あなたはスプラトゥーンのゲーム画面の上部から試合タイマーを読み取る専門AIです。
この画像はゲーム画面の上半分のみをクロップしたものです。

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

TIMER_SCAN_USER_PROMPT = "この画像（ゲーム画面の上半分）から試合タイマーの残り時間を読み取ってJSON形式で回答してください。"

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
class _TimerReading:
    """Internal: a single timer reading from one frame."""

    frame_timestamp: float
    timer_seconds: float
    total_duration: int
    duration_type: str
    match_start: float


def cluster_readings(readings: list[_TimerReading]) -> list[MatchInfo]:
    """Group timer readings into matches by clustering estimated start times.

    Args:
        readings: Sorted list of timer readings by frame_timestamp.

    Returns:
        List of MatchInfo sorted by start_seconds.
    """
    if not readings:
        return []

    sorted_readings = sorted(readings, key=lambda r: r.match_start)

    clusters: list[list[_TimerReading]] = []
    current_cluster: list[_TimerReading] = [sorted_readings[0]]

    for reading in sorted_readings[1:]:
        if abs(reading.match_start - current_cluster[-1].match_start) <= CLUSTER_TOLERANCE:
            current_cluster.append(reading)
        else:
            clusters.append(current_cluster)
            current_cluster = [reading]
    clusters.append(current_cluster)

    matches: list[MatchInfo] = []
    for cluster in clusters:
        starts = [r.match_start for r in cluster]
        median_start = statistics.median(starts)

        # Use the most common duration type in the cluster
        type_counts: dict[str, int] = {}
        for r in cluster:
            type_counts[r.duration_type] = type_counts.get(r.duration_type, 0) + 1
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
    return matches


class MatchScanner:
    """Scan a video recording to detect individual match boundaries."""

    def __init__(self, analyzer: BattleAnalyzer, interval: float = 30.0) -> None:
        """Initialize the scanner.

        Args:
            analyzer: BattleAnalyzer instance for frame analysis.
            interval: Frame extraction interval in seconds (default: 30).
        """
        self.analyzer = analyzer
        self.interval = interval

    def scan(
        self,
        video_path: str | Path,
        progress_callback: ScanProgressCallback | None = None,
    ) -> list[MatchInfo]:
        """Scan a video for match boundaries.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback (frames_done, frames_total).

        Returns:
            List of detected matches sorted by start_seconds.
        """
        video_path = Path(video_path)

        frames = extract_frames(
            video_path=video_path,
            interval_seconds=self.interval,
            no_save=True,
        )

        total_frames = len(frames)
        readings: list[_TimerReading | None] = [None] * total_frames
        done_count = [0]

        def _analyze_one(index: int) -> None:
            timestamp_sec = index * self.interval
            ts_label = self._format_timestamp(timestamp_sec)

            frame = frames[index]
            h = frame.shape[0]
            upper_half = frame[: h // 2, :, :]

            try:
                result = self.analyzer._analyze_cropped(
                    upper_half,
                    TIMER_SCAN_USER_PROMPT,
                    TIMER_SCAN_SYSTEM_PROMPT,
                    ts_label,
                )
            except Exception:
                logger.exception("Timer scan failed for frame at %s", ts_label)
                result = {}

            timer_value = None
            if isinstance(result, dict):
                raw_timer = result.get("timer_remaining")
                if raw_timer and isinstance(raw_timer, str):
                    timer_value = parse_timer(raw_timer)

            if timer_value is not None:
                total_duration, duration_type = determine_rule(timer_value)
                match_start = calc_match_start(timestamp_sec, timer_value, total_duration)
                readings[index] = _TimerReading(
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

        return cluster_readings(valid_readings)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"
