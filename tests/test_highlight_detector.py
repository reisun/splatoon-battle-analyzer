"""Tests for the highlight detection module."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.highlight_detector import (
    MAX_CLIP_SECONDS,
    HighlightDetector,
    HighlightSegment,
    _cap_segment_duration,
    _compute_score,
)


class TestComputeScore:
    """Tests for _compute_score helper."""

    def test_all_ones(self) -> None:
        result = {"kills": 1, "assists": 1, "score_gain": 1, "special": 1}
        assert _compute_score(result) == 1

    def test_all_tens(self) -> None:
        result = {"kills": 10, "assists": 10, "score_gain": 10, "special": 10}
        assert _compute_score(result) == 10000

    def test_mixed_values(self) -> None:
        result = {"kills": 5, "assists": 2, "score_gain": 3, "special": 4}
        assert _compute_score(result) == 5 * 2 * 3 * 4

    def test_missing_keys_default_to_one(self) -> None:
        assert _compute_score({}) == 1

    def test_clamps_below_one(self) -> None:
        result = {"kills": 0, "assists": -5, "score_gain": 1, "special": 1}
        assert _compute_score(result) == 1

    def test_clamps_above_ten(self) -> None:
        result = {"kills": 99, "assists": 1, "score_gain": 1, "special": 1}
        assert _compute_score(result) == 10


class TestCapSegmentDuration:
    """Tests for _cap_segment_duration helper."""

    def test_short_segment_unchanged(self) -> None:
        seg = HighlightSegment(
            start_seconds=10.0, end_seconds=20.0, peak_intensity=100, description="x"
        )
        result = _cap_segment_duration(seg)
        assert result.start_seconds == 10.0
        assert result.end_seconds == 20.0

    def test_long_segment_capped(self) -> None:
        seg = HighlightSegment(
            start_seconds=0.0, end_seconds=30.0, peak_intensity=100, description="x"
        )
        result = _cap_segment_duration(seg)
        assert result.end_seconds - result.start_seconds == MAX_CLIP_SECONDS

    def test_cap_centers_on_midpoint(self) -> None:
        seg = HighlightSegment(
            start_seconds=10.0, end_seconds=40.0, peak_intensity=100, description="x"
        )
        result = _cap_segment_duration(seg)
        center = (10.0 + 40.0) / 2
        assert result.start_seconds == center - MAX_CLIP_SECONDS / 2
        assert result.end_seconds == center + MAX_CLIP_SECONDS / 2

    def test_cap_does_not_go_below_zero(self) -> None:
        seg = HighlightSegment(
            start_seconds=0.0, end_seconds=20.0, peak_intensity=100, description="x"
        )
        result = _cap_segment_duration(seg)
        assert result.start_seconds >= 0


class TestHighlightSegment:
    """Tests for HighlightSegment dataclass."""

    def test_creation(self) -> None:
        seg = HighlightSegment(
            start_seconds=10.0, end_seconds=25.0, peak_intensity=800, description="test"
        )
        assert seg.start_seconds == 10.0
        assert seg.end_seconds == 25.0
        assert seg.peak_intensity == 800
        assert seg.description == "test"


class TestHighlightDetectorInit:
    """Tests for HighlightDetector initialization."""

    def test_defaults(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        assert detector.stage1_interval == 30
        assert detector.stage2_interval == 3
        assert detector.threshold == 100
        assert detector.max_highlights == 4

    def test_custom_params(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(
            analyzer=analyzer, stage1_interval=60, stage2_interval=10, threshold=200
        )
        assert detector.stage1_interval == 60
        assert detector.stage2_interval == 10
        assert detector.threshold == 200


class TestBuildRegions:
    """Tests for region building logic."""

    def test_single_candidate(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30)
        regions = detector._build_regions([90.0], None, None)
        assert regions == [(60.0, 120.0)]

    def test_overlapping_candidates_merge(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30)
        regions = detector._build_regions([60.0, 90.0], None, None)
        assert regions == [(30.0, 120.0)]

    def test_non_overlapping_candidates(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30)
        regions = detector._build_regions([60.0, 180.0], None, None)
        assert len(regions) == 2
        assert regions[0] == (30.0, 90.0)
        assert regions[1] == (150.0, 210.0)

    def test_respects_start_bound(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30)
        regions = detector._build_regions([10.0], 5.0, None)
        assert regions[0][0] == 5.0

    def test_respects_end_bound(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30)
        regions = detector._build_regions([100.0], None, 110.0)
        assert regions[0][1] == 110.0


class TestMergeSegments:
    """Tests for segment merging logic."""

    def test_empty_results(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        segments = detector._merge_segments([])
        assert segments == []

    def test_single_high_score_frame(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=3, threshold=100)
        # score = 5*4*2*3 = 120, above threshold 100
        results = [
            (
                10.0,
                {
                    "kills": 5,
                    "assists": 4,
                    "score_gain": 2,
                    "special": 3,
                    "description": "action",
                },
            )
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 1
        assert segments[0].start_seconds == 10.0
        assert segments[0].end_seconds == 13.0
        assert segments[0].peak_intensity == 120

    def test_consecutive_frames_merge(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=3, threshold=100)
        # All scores above 100
        results = [
            (
                10.0,
                {
                    "kills": 5,
                    "assists": 4,
                    "score_gain": 2,
                    "special": 3,
                    "description": "start",
                },
            ),
            (
                13.0,
                {
                    "kills": 8,
                    "assists": 4,
                    "score_gain": 2,
                    "special": 3,
                    "description": "peak",
                },
            ),
            (
                16.0,
                {
                    "kills": 5,
                    "assists": 4,
                    "score_gain": 2,
                    "special": 3,
                    "description": "end",
                },
            ),
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 1
        assert segments[0].start_seconds == 10.0
        assert segments[0].end_seconds == 19.0
        assert segments[0].peak_intensity == 8 * 4 * 2 * 3

    def test_low_score_splits_segments(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=3, threshold=100)
        results = [
            (
                10.0,
                {
                    "kills": 5,
                    "assists": 4,
                    "score_gain": 2,
                    "special": 3,
                    "description": "first",
                },
            ),
            (
                13.0,
                {
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "description": "low",
                },
            ),
            (
                16.0,
                {
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "description": "low",
                },
            ),
            (
                19.0,
                {
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "description": "low",
                },
            ),
            (
                22.0,
                {
                    "kills": 8,
                    "assists": 4,
                    "score_gain": 2,
                    "special": 3,
                    "description": "second",
                },
            ),
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 2

    def test_non_dict_result_treated_as_zero(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=3, threshold=100)
        results = [
            (
                10.0,
                {
                    "kills": 5,
                    "assists": 4,
                    "score_gain": 2,
                    "special": 3,
                    "description": "action",
                },
            ),
            (13.0, "parse error"),
            (16.0, "parse error"),
            (19.0, "parse error"),
            (
                22.0,
                {
                    "kills": 5,
                    "assists": 4,
                    "score_gain": 2,
                    "special": 3,
                    "description": "more action",
                },
            ),
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 2


class TestDetectFlow:
    """Tests for the full detect() pipeline."""

    @patch("src.highlight_detector.extract_frames")
    def test_no_candidates_returns_empty(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "scene": "lobby",
            "kills": 1,
            "assists": 1,
            "score_gain": 1,
            "special": 1,
            "reason": "lobby screen",
        }

        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30, threshold=100)
        segments = detector.detect("/fake/video.mp4")

        assert segments == []
        assert detector.stage1_summary["total_frames"] == 3
        assert detector.stage1_summary["battle_frames"] == 0
        assert detector.stage1_summary["candidate_frames"] == 0

    @patch("src.highlight_detector.extract_frames")
    def test_full_pipeline(self, mock_extract: MagicMock) -> None:
        stage1_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        stage2_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        mock_extract.side_effect = [stage1_frames, stage2_frames]

        call_count = [0]

        def mock_analyze(frame, prompt, timestamp):
            call_count[0] += 1
            from src.battle_analyzer import STAGE1_PROMPT

            if prompt == STAGE1_PROMPT:
                if call_count[0] == 2:
                    return {
                        "scene": "battle",
                        "kills": 5,
                        "assists": 4,
                        "score_gain": 2,
                        "special": 3,
                        "reason": "intense fight",
                    }
                return {
                    "scene": "battle",
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "reason": "calm",
                }
            return {
                "kills": 5,
                "assists": 4,
                "score_gain": 2,
                "special": 3,
                "description": "kills happening",
            }

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        detector = HighlightDetector(
            analyzer=analyzer, stage1_interval=30, stage2_interval=3, threshold=100
        )
        segments = detector.detect("/fake/video.mp4")

        assert len(segments) >= 1
        assert detector.stage1_summary["total_frames"] == 4
        # All 4 are battle frames, but only 1 has score > 1
        # Actually all are candidates (score>=1), top 4 selected
        assert detector.stage1_summary["candidate_frames"] == 4

    @patch("src.highlight_detector.extract_frames")
    def test_progress_callback_called(self, mock_extract: MagicMock) -> None:
        """Progress callback is invoked for each stage."""
        stage1_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3
        mock_extract.return_value = stage1_frames

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "scene": "lobby",
            "kills": 1,
            "assists": 1,
            "score_gain": 1,
            "special": 1,
            "reason": "lobby screen",
        }

        progress_calls: list[tuple[int, int, int]] = []

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            progress_calls.append((phase, frames_done, frames_total))

        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30, threshold=100)
        detector.detect("/fake/video.mp4", progress_callback=on_progress)

        # Only stage 1 runs (no candidates for stage 2)
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 1, 3)
        assert progress_calls[1] == (1, 2, 3)
        assert progress_calls[2] == (1, 3, 3)

    @patch("src.highlight_detector.extract_frames")
    def test_progress_callback_stage2(self, mock_extract: MagicMock) -> None:
        """Progress callback reports both stage 1 and stage 2 progress."""
        stage1_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        stage2_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        mock_extract.side_effect = [stage1_frames, stage2_frames]

        call_count = [0]

        def mock_analyze(frame, prompt, timestamp):
            call_count[0] += 1
            from src.battle_analyzer import STAGE1_PROMPT

            if prompt == STAGE1_PROMPT:
                if call_count[0] == 2:
                    return {
                        "scene": "battle",
                        "kills": 5,
                        "assists": 4,
                        "score_gain": 2,
                        "special": 3,
                        "reason": "intense fight",
                    }
                return {
                    "scene": "battle",
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "reason": "calm",
                }
            return {
                "kills": 5,
                "assists": 4,
                "score_gain": 2,
                "special": 3,
                "description": "kills happening",
            }

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        progress_calls: list[tuple[int, int, int]] = []

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            progress_calls.append((phase, frames_done, frames_total))

        detector = HighlightDetector(
            analyzer=analyzer, stage1_interval=30, stage2_interval=3, threshold=100
        )
        detector.detect("/fake/video.mp4", progress_callback=on_progress)

        stage1_calls = [c for c in progress_calls if c[0] == 1]
        stage2_calls = [c for c in progress_calls if c[0] == 2]
        assert len(stage1_calls) == 4
        assert len(stage2_calls) == 3
        # Stage 2 final call should have frames_done == frames_total
        assert stage2_calls[-1][1] == stage2_calls[-1][2]

    @patch("src.highlight_detector.extract_frames")
    def test_battle_with_low_score_not_highlighted(self, mock_extract: MagicMock) -> None:
        """Battle frames with all-1 scores still become candidates but don't pass threshold."""
        stage1_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 2
        stage2_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        mock_extract.side_effect = [stage1_frames, stage2_frames]

        def mock_analyze(frame, prompt, timestamp):
            from src.battle_analyzer import STAGE1_PROMPT

            if prompt == STAGE1_PROMPT:
                return {
                    "scene": "battle",
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "reason": "low action",
                }
            return {
                "kills": 1,
                "assists": 1,
                "score_gain": 1,
                "special": 1,
                "description": "nothing happening",
            }

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30, threshold=100)
        segments = detector.detect("/fake/video.mp4")

        # All battle frames are candidates (score=1), but stage2 score=1 < threshold=100
        assert segments == []
        assert detector.stage1_summary["battle_frames"] == 2
        assert detector.stage1_summary["candidate_frames"] == 2
