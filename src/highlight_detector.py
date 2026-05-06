"""Two-stage highlight detection for Splatoon gameplay videos."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.battle_analyzer import STAGE1_PROMPT, STAGE2_PROMPT, BattleAnalyzer
from src.frame_extractor import extract_frames

ProgressCallback = Callable[[int, int, int], None]  # (phase, frames_done, frames_total)

logger = logging.getLogger(__name__)

MAX_CLIP_SECONDS = 15
MAX_TOTAL_SECONDS = 60
WINDOW_SIZE = 5  # 5 frames * 3s interval = 15s


def _compute_frame_score(result: dict) -> int:
    """観測値からスコアを計算。"""
    kills = result.get("kills_in_log", 0)
    assists = result.get("assists_in_log", 0)
    team_score_increasing = result.get("team_score_increasing", False)
    my_special_active = result.get("my_special_active", False)
    is_dead = result.get("is_dead", False)

    # 各項目を1-10にマッピング
    kills_score = min(10, 1 + kills * 3)  # 0->1, 1->4, 2->7, 3+->10
    assists_score = min(10, 1 + assists * 3)  # 同上
    score_gain = 5 if team_score_increasing else 1
    special = 10 if my_special_active else 1

    total = kills_score * assists_score * score_gain * special

    # やられている場合はスコア半減
    if is_dead:
        total = total // 2

    return max(1, total)


def _compute_stage1_score(result: dict) -> int:
    """Stage1用の簡易スコア。"""
    kills = result.get("kills_in_log", 0)
    my_special_active = result.get("my_special_active", False)
    is_dead = result.get("is_dead", False)

    kills_score = min(10, 1 + kills * 3)
    special = 10 if my_special_active else 1

    total = kills_score * special

    if is_dead:
        total = total // 2

    return max(1, total)


def _cap_segment_duration(segment: HighlightSegment) -> HighlightSegment:
    """Cap a segment to MAX_CLIP_SECONDS, centered on the midpoint."""
    duration = segment.end_seconds - segment.start_seconds
    if duration > MAX_CLIP_SECONDS:
        center = (segment.start_seconds + segment.end_seconds) / 2
        segment.start_seconds = max(0, center - MAX_CLIP_SECONDS / 2)
        segment.end_seconds = center + MAX_CLIP_SECONDS / 2
    return segment


@dataclass
class FrameAnalysis:
    timestamp_seconds: float
    kills_in_log: int = 0
    assists_in_log: int = 0
    team_score_increasing: bool = False
    my_special_active: bool = False
    is_dead: bool = False
    score: int = 0
    description: str = ""


@dataclass
class HighlightSegment:
    start_seconds: float
    end_seconds: float
    peak_intensity: int
    description: str
    frames: list[FrameAnalysis] = field(default_factory=list)


class HighlightDetector:
    """Detect highlight segments using a two-stage analysis pipeline."""

    def __init__(
        self,
        analyzer: BattleAnalyzer,
        stage1_interval: float = 30,
        stage2_interval: float = 3,
        threshold: int = 100,
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
                result = {
                    "scene": "other",
                    "kills_in_log": 0,
                    "reason": "analysis failed",
                }
            stage1_results.append((timestamp_sec, result))

            if isinstance(result, dict):
                scene = result.get("scene", "other")
                if scene == "battle":
                    battle_count += 1
                    score = _compute_stage1_score(result)
                    candidates.append((timestamp_sec, score))

            if progress_callback:
                progress_callback(1, i + 1, len(stage1_frames))

        # Select top N by score
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

        frame_analyses: list[FrameAnalysis] = []
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
                    result = {"kills_in_log": 0, "description": "analysis failed"}

                score = _compute_frame_score(result) if isinstance(result, dict) else 1

                fa = FrameAnalysis(
                    timestamp_seconds=timestamp_sec,
                    kills_in_log=(
                        result.get("kills_in_log", 0) if isinstance(result, dict) else 0
                    ),
                    assists_in_log=(
                        result.get("assists_in_log", 0) if isinstance(result, dict) else 0
                    ),
                    team_score_increasing=(
                        result.get("team_score_increasing", False)
                        if isinstance(result, dict)
                        else False
                    ),
                    my_special_active=(
                        result.get("my_special_active", False)
                        if isinstance(result, dict)
                        else False
                    ),
                    is_dead=(
                        result.get("is_dead", False) if isinstance(result, dict) else False
                    ),
                    score=score,
                    description=(
                        result.get("description", "") if isinstance(result, dict) else ""
                    ),
                )
                frame_analyses.append(fa)

                stage2_done += 1
                if progress_callback:
                    progress_callback(2, stage2_done, total_stage2_frames)

        # Sliding window detection (replaces _merge_segments)
        segments = self._find_best_windows(frame_analyses)

        # Cap each segment duration and enforce total time limit
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

    def _find_best_windows(
        self,
        frame_analyses: list[FrameAnalysis],
    ) -> list[HighlightSegment]:
        """スライディングウィンドウで最高スコアの区間を検出。"""
        if len(frame_analyses) < WINDOW_SIZE:
            # フレームが5未満の場合は全フレームを1つのセグメントに
            if not frame_analyses:
                return []
            total_score = sum(f.score for f in frame_analyses)
            if total_score < self.threshold:
                return []
            return [
                HighlightSegment(
                    start_seconds=frame_analyses[0].timestamp_seconds,
                    end_seconds=(
                        frame_analyses[-1].timestamp_seconds + self.stage2_interval
                    ),
                    peak_intensity=total_score,
                    description=self._summarize_descriptions(
                        [f.description for f in frame_analyses]
                    ),
                    frames=list(frame_analyses),
                )
            ]

        # 各ウィンドウのスコアを計算
        windows: list[tuple[int, int, list[FrameAnalysis]]] = []
        for i in range(len(frame_analyses) - WINDOW_SIZE + 1):
            window_frames = frame_analyses[i : i + WINDOW_SIZE]
            window_score = sum(f.score for f in window_frames)
            windows.append((window_score, i, window_frames))

        # スコア降順にソート
        windows.sort(key=lambda x: x[0], reverse=True)

        # 重複しない上位N個を選択
        selected: list[HighlightSegment] = []
        used_indices: set[int] = set()

        for window_score, start_idx, window_frames in windows:
            if len(selected) >= self.max_highlights:
                break

            # threshold以下はスキップ
            if window_score < self.threshold:
                break

            # 重複チェック
            window_indices = set(range(start_idx, start_idx + WINDOW_SIZE))
            if window_indices & used_indices:
                continue

            used_indices |= window_indices

            segment = HighlightSegment(
                start_seconds=window_frames[0].timestamp_seconds,
                end_seconds=(
                    window_frames[-1].timestamp_seconds + self.stage2_interval
                ),
                peak_intensity=window_score,
                description=self._summarize_descriptions(
                    [f.description for f in window_frames]
                ),
                frames=list(window_frames),
            )
            selected.append(segment)

        # 時系列順にソート
        selected.sort(key=lambda s: s.start_seconds)
        return selected

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
