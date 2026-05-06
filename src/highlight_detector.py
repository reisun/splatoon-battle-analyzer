"""Single-pass highlight detection for Splatoon gameplay videos."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.battle_analyzer import STAGE2_PROMPT, BattleAnalyzer
from src.frame_extractor import extract_frames

ProgressCallback = Callable[[int, int, int], None]  # (phase, frames_done, frames_total)

logger = logging.getLogger(__name__)

MAX_CLIP_SECONDS = 15
MAX_TOTAL_SECONDS = 60


def _compute_score(result: dict) -> int:
    """5項目の掛け算でスコアを計算（1-100000）."""
    kills = max(1, min(10, result.get("kills", 1)))
    assists = max(1, min(10, result.get("assists", 1)))
    score_gain = max(1, min(10, result.get("score_gain", 1)))
    clutch = max(1, min(10, result.get("clutch", 1)))
    special = max(1, min(10, result.get("special", 1)))
    return kills * assists * score_gain * clutch * special


def _cap_segment_duration(segment: "HighlightSegment") -> "HighlightSegment":
    """Cap a segment to MAX_CLIP_SECONDS, centered on the midpoint."""
    duration = segment.end_seconds - segment.start_seconds
    if duration > MAX_CLIP_SECONDS:
        center = (segment.start_seconds + segment.end_seconds) / 2
        segment.start_seconds = max(0, center - MAX_CLIP_SECONDS / 2)
        segment.end_seconds = center + MAX_CLIP_SECONDS / 2
    return segment


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
                    frames[index], STAGE2_PROMPT, ts_label
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

        segments = self._merge_segments(results)
        segments = [s for s in segments if s.peak_intensity >= self.threshold]

        segments = [_cap_segment_duration(s) for s in segments]
        total = sum(s.end_seconds - s.start_seconds for s in segments)
        if total > MAX_TOTAL_SECONDS:
            segments.sort(key=lambda s: s.peak_intensity, reverse=True)
            result_segments: list[HighlightSegment] = []
            remaining = MAX_TOTAL_SECONDS
            for s in segments:
                dur = s.end_seconds - s.start_seconds
                if dur <= remaining:
                    result_segments.append(s)
                    remaining -= dur
            segments = sorted(result_segments, key=lambda s: s.start_seconds)

        return segments

    def _merge_segments(
        self, results: list[tuple[float, dict | str]]
    ) -> list[HighlightSegment]:
        """Merge consecutive high-score frames into highlight segments."""
        if not results:
            return []

        sorted_results = sorted(results, key=lambda x: x[0])
        segments: list[HighlightSegment] = []
        current_start: float | None = None
        current_end: float = 0
        current_peak: int = 0
        current_descriptions: list[str] = []

        for timestamp, result in sorted_results:
            score = 0
            description = ""
            if isinstance(result, dict):
                score = _compute_score(result)
                description = result.get("description", "")

            if score >= self.threshold:
                if current_start is None:
                    current_start = timestamp
                current_end = timestamp
                current_peak = max(current_peak, score)
                if description:
                    current_descriptions.append(description)
            else:
                if current_start is not None:
                    gap = timestamp - current_end
                    if gap <= self.interval:
                        continue
                    segments.append(
                        HighlightSegment(
                            start_seconds=current_start,
                            end_seconds=current_end + self.interval,
                            peak_intensity=current_peak,
                            description=self._summarize_descriptions(current_descriptions),
                        )
                    )
                    current_start = None
                    current_end = 0
                    current_peak = 0
                    current_descriptions = []

        if current_start is not None:
            segments.append(
                HighlightSegment(
                    start_seconds=current_start,
                    end_seconds=current_end + self.interval,
                    peak_intensity=current_peak,
                    description=self._summarize_descriptions(current_descriptions),
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
