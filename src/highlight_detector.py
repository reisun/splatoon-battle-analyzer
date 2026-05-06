"""Single-pass highlight detection for Splatoon gameplay videos."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.battle_analyzer import FRAME_ANALYSIS_PROMPT, BattleAnalyzer
from src.frame_extractor import extract_frames

ProgressCallback = Callable[[int, int, int], None]  # (phase, frames_done, frames_total)

logger = logging.getLogger(__name__)

MAX_CLIP_SECONDS = 15
MAX_TOTAL_SECONDS = 60


def _calc_score_gain(prev_count: int | None, cur_count: int | None) -> int:
    """前フレームとのゲームカウント差分から score_gain を計算."""
    if prev_count is None or cur_count is None:
        return 1
    gain = (prev_count - cur_count) / 10 + 1
    return max(1, min(10, int(gain)))


def _compute_score(result: dict) -> int:
    """4項目の掛け算でスコアを計算。デス中は半減."""
    kills = max(1, min(10, result.get("kills", 1)))
    assists = max(1, min(10, result.get("assists", 1)))
    score_gain = max(1, min(10, result.get("score_gain", 1)))
    special = max(1, min(10, result.get("special", 1)))
    score = kills * assists * score_gain * special
    if result.get("is_dead", False):
        score //= 2
    return score


@dataclass
class _ScoredFrame:
    timestamp: float
    score: int
    description: str
    raw: dict


@dataclass
class FrameAnalysis:
    timestamp_seconds: float
    score: int
    kills: int
    assists: int
    score_gain: int
    special: int
    is_dead: bool
    description: str
    my_team_color: str
    enemy_team_color: str
    my_team_count: int | None
    enemy_team_count: int | None


@dataclass
class HighlightSegment:
    start_seconds: float
    end_seconds: float
    peak_intensity: int
    description: str


class HighlightDetector:
    """Detect highlight segments using single-pass parallel analysis."""

    def __init__(
        self,
        analyzer: BattleAnalyzer,
        interval: float = 5,
        threshold: int = 100,
    ) -> None:
        self.analyzer = analyzer
        self.interval = interval
        self.threshold = threshold
        self.scan_summary: dict = {}
        self.all_frames: list[FrameAnalysis] = []

    def detect(
        self,
        video_path: str | Path,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[HighlightSegment]:
        video_path = Path(video_path)

        frames = extract_frames(
            video_path=video_path,
            interval_seconds=self.interval,
            no_save=True,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

        scan_start = start_seconds or 0.0
        total_frames = len(frames)
        results: list[tuple[float, dict | str]] = [None] * total_frames  # type: ignore[list-item]
        done_count = [0]

        def _analyze_one(index: int) -> None:
            timestamp_sec = scan_start + index * self.interval
            ts_label = self._format_timestamp(timestamp_sec)
            try:
                result = self.analyzer.analyze_frame_from_memory_with_prompt(
                    frames[index], FRAME_ANALYSIS_PROMPT, ts_label
                )
            except Exception:
                logger.exception("Analysis failed for frame at %s", ts_label)
                result = {"kills": 1, "description": "analysis failed"}
            results[index] = (timestamp_sec, result)

            done_count[0] += 1
            if progress_callback:
                progress_callback(1, done_count[0], total_frames)

        with ThreadPoolExecutor(max_workers=self.analyzer.concurrency) as executor:
            futures = [executor.submit(_analyze_one, i) for i in range(total_frames)]
            for future in as_completed(futures):
                future.result()

        battle_count = sum(
            1 for _, r in results if isinstance(r, dict) and r.get("scene") != "other"
        )
        self.scan_summary = {
            "total_frames": total_frames,
            "battle_frames": battle_count,
        }

        scored = self._score_frames(results)
        self.all_frames = [
            FrameAnalysis(
                timestamp_seconds=f.timestamp,
                score=f.score,
                kills=max(1, min(10, f.raw.get("kills", 1))),
                assists=max(1, min(10, f.raw.get("assists", 1))),
                score_gain=max(1, min(10, f.raw.get("score_gain", 1))),
                special=max(1, min(10, f.raw.get("special", 1))),
                is_dead=f.raw.get("is_dead", False),
                description=f.raw.get("description", ""),
                my_team_color=f.raw.get("my_team_color", ""),
                enemy_team_color=f.raw.get("enemy_team_color", ""),
                my_team_count=f.raw.get("my_team_count"),
                enemy_team_count=f.raw.get("enemy_team_count"),
            )
            for f in scored
        ]
        segments = self._select_windows(scored)

        remaining = MAX_TOTAL_SECONDS
        budget_segments: list[HighlightSegment] = []
        for s in segments:
            dur = s.end_seconds - s.start_seconds
            if dur <= remaining:
                budget_segments.append(s)
                remaining -= dur

        return budget_segments

    def _score_frames(self, results: list[tuple[float, dict | str]]) -> list[_ScoredFrame]:
        sorted_results = sorted(results, key=lambda x: x[0])
        scored: list[_ScoredFrame] = []
        prev_count: int | None = None
        for timestamp, result in sorted_results:
            score = 0
            description = ""
            raw: dict = {}
            if isinstance(result, dict):
                raw = result
                cur_count = raw.get("my_team_count")
                raw["score_gain"] = _calc_score_gain(prev_count, cur_count)
                if cur_count is not None:
                    prev_count = cur_count
                score = _compute_score(raw)
                description = raw.get("description", "")
            scored.append(_ScoredFrame(timestamp, score, description, raw))
        return scored

    def _select_windows(self, scored: list[_ScoredFrame]) -> list[HighlightSegment]:
        """Select best non-overlapping windows by sliding window score sum."""
        if not scored:
            return []

        window_size = max(1, int(MAX_CLIP_SECONDS / self.interval))
        n = len(scored)

        windows: list[tuple[int, int, int]] = []
        for i in range(n - window_size + 1):
            window = scored[i : i + window_size]
            total = sum(f.score for f in window)
            peak = max(f.score for f in window)
            if peak >= self.threshold:
                windows.append((i, total, peak))

        if not windows:
            for i, frame in enumerate(scored):
                if frame.score >= self.threshold:
                    windows.append((i, frame.score, frame.score))

        if not windows:
            return []

        windows.sort(key=lambda w: w[1], reverse=True)

        selected: list[int] = []
        used: set[int] = set()
        for start_idx, _total, _peak in windows:
            end_idx = min(start_idx + window_size - 1, n - 1)
            frame_indices = set(range(start_idx, end_idx + 1))
            if frame_indices & used:
                continue
            selected.append(start_idx)
            used.update(frame_indices)

        selected.sort()

        segments: list[HighlightSegment] = []
        for start_idx in selected:
            end_idx = min(start_idx + window_size - 1, n - 1)
            window = scored[start_idx : end_idx + 1]
            descriptions = [f.description for f in window if f.description]
            segments.append(
                HighlightSegment(
                    start_seconds=scored[start_idx].timestamp,
                    end_seconds=scored[end_idx].timestamp + self.interval,
                    peak_intensity=sum(f.score for f in window),
                    description=self._summarize_descriptions(descriptions),
                )
            )

        return segments

    @staticmethod
    def _summarize_descriptions(descriptions: list[str]) -> str:
        if not descriptions:
            return ""
        unique = list(dict.fromkeys(descriptions))
        return "; ".join(unique[:3])

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"
