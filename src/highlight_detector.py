"""Two-stage highlight detection for Splatoon gameplay videos."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.battle_analyzer import STAGE1_PROMPT, STAGE2_PROMPT, BattleAnalyzer
from src.frame_extractor import extract_frames

ProgressCallback = Callable[[int, int, int], None]  # (phase, frames_done, frames_total)

logger = logging.getLogger(__name__)


@dataclass
class HighlightSegment:
    start_seconds: float
    end_seconds: float
    peak_intensity: int
    description: str


class HighlightDetector:
    """Detect highlight segments using a two-stage analysis pipeline."""

    def __init__(
        self,
        analyzer: BattleAnalyzer,
        stage1_interval: float = 30,
        stage2_interval: float = 5,
        threshold: int = 5,
        max_highlights: int = 3,
    ) -> None:
        self.analyzer = analyzer
        self.stage1_interval = stage1_interval
        self.stage2_interval = stage2_interval
        self.threshold = threshold
        self.max_highlights = max_highlights
        self.stage1_summary: dict = {}

    def detect(
        self,
        video_path: str | Path,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[HighlightSegment]:
        video_path = Path(video_path)

        # Stage 1: coarse scan
        stage1_frames = extract_frames(
            video_path=video_path,
            interval_seconds=self.stage1_interval,
            no_save=True,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

        stage1_start = start_seconds or 0.0
        stage1_results: list[tuple[float, dict | str]] = []
        battle_count = 0
        candidates: list[tuple[float, int]] = []

        for i, frame in enumerate(stage1_frames):
            timestamp_sec = stage1_start + i * self.stage1_interval
            ts_label = self._format_timestamp(timestamp_sec)
            try:
                result = self.analyzer.analyze_frame_from_memory_with_prompt(
                    frame, STAGE1_PROMPT, ts_label
                )
            except Exception:
                logger.exception("Stage 1 failed for frame at %s", ts_label)
                result = {"scene": "other", "intensity": 0, "reason": "analysis failed"}
            stage1_results.append((timestamp_sec, result))

            if isinstance(result, dict):
                scene = result.get("scene", "other")
                intensity = result.get("intensity", 0)
                if scene == "battle":
                    battle_count += 1
                    if intensity >= self.threshold:
                        candidates.append((timestamp_sec, intensity))

            if progress_callback:
                progress_callback(1, i + 1, len(stage1_frames))

        # Select top N by intensity
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[: self.max_highlights]
        candidate_timestamps = [ts for ts, _ in top_candidates]

        self.stage1_summary = {
            "total_frames": len(stage1_frames),
            "battle_frames": battle_count,
            "candidate_frames": len(candidate_timestamps),
        }

        if not candidate_timestamps:
            return []

        # Stage 2: fine-grained analysis of candidate regions
        regions = self._build_regions(candidate_timestamps, start_seconds, end_seconds)
        stage2_results: list[tuple[float, dict | str]] = []

        # Pre-extract all Stage 2 frames to know total count for progress
        all_stage2_frames: list[tuple[float, list]] = []
        total_stage2_frames = 0
        for region_start, region_end in regions:
            frames = extract_frames(
                video_path=video_path,
                interval_seconds=self.stage2_interval,
                no_save=True,
                start_seconds=region_start,
                end_seconds=region_end,
            )
            all_stage2_frames.append((region_start, frames))
            total_stage2_frames += len(frames)

        stage2_done = 0
        for region_start, frames in all_stage2_frames:
            for i, frame in enumerate(frames):
                timestamp_sec = region_start + i * self.stage2_interval
                ts_label = self._format_timestamp(timestamp_sec)
                try:
                    result = self.analyzer.analyze_frame_from_memory_with_prompt(
                        frame, STAGE2_PROMPT, ts_label
                    )
                except Exception:
                    logger.exception("Stage 2 failed for frame at %s", ts_label)
                    result = {"intensity": 0, "description": "analysis failed"}
                stage2_results.append((timestamp_sec, result))

                stage2_done += 1
                if progress_callback:
                    progress_callback(2, stage2_done, total_stage2_frames)

        # Merge into segments and filter by threshold
        segments = self._merge_segments(stage2_results)
        return [s for s in segments if s.peak_intensity >= self.threshold]

    def _build_regions(
        self,
        candidates: list[float],
        start_seconds: float | None,
        end_seconds: float | None,
    ) -> list[tuple[float, float]]:
        """Expand candidate timestamps into regions and merge overlapping ones."""
        margin = self.stage1_interval
        min_start = start_seconds or 0.0
        max_end = end_seconds  # None means no upper bound from user

        raw_regions = []
        for ts in candidates:
            r_start = max(min_start, ts - margin)
            r_end = ts + margin
            if max_end is not None:
                r_end = min(max_end, r_end)
            raw_regions.append((r_start, r_end))

        if not raw_regions:
            return []

        raw_regions.sort()
        merged = [raw_regions[0]]
        for start, end in raw_regions[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return merged

    def _merge_segments(
        self, results: list[tuple[float, dict | str]]
    ) -> list[HighlightSegment]:
        """Merge consecutive high-intensity frames into highlight segments."""
        if not results:
            return []

        results.sort(key=lambda x: x[0])
        segments: list[HighlightSegment] = []
        current_start: float | None = None
        current_end: float = 0
        current_peak: int = 0
        current_descriptions: list[str] = []

        for timestamp, result in results:
            intensity = 0
            description = ""
            if isinstance(result, dict):
                intensity = result.get("intensity", 0)
                description = result.get("description", "")

            if intensity >= self.threshold:
                if current_start is None:
                    current_start = timestamp
                current_end = timestamp
                current_peak = max(current_peak, intensity)
                if description:
                    current_descriptions.append(description)
            else:
                if current_start is not None:
                    gap = timestamp - current_end
                    if gap <= self.stage2_interval:
                        continue
                    segments.append(
                        HighlightSegment(
                            start_seconds=current_start,
                            end_seconds=current_end + self.stage2_interval,
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
                    end_seconds=current_end + self.stage2_interval,
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
