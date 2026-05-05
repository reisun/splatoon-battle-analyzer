"""Tests for the highlight detection module."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.highlight_detector import HighlightDetector, HighlightSegment


class TestHighlightSegment:
    """Tests for HighlightSegment dataclass."""

    def test_creation(self) -> None:
        seg = HighlightSegment(
            start_seconds=10.0, end_seconds=25.0, peak_intensity=8, description="test"
        )
        assert seg.start_seconds == 10.0
        assert seg.end_seconds == 25.0
        assert seg.peak_intensity == 8
        assert seg.description == "test"


class TestHighlightDetectorInit:
    """Tests for HighlightDetector initialization."""

    def test_defaults(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        assert detector.stage1_interval == 30
        assert detector.stage2_interval == 5
        assert detector.threshold == 5

    def test_custom_params(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(
            analyzer=analyzer, stage1_interval=60, stage2_interval=10, threshold=7
        )
        assert detector.stage1_interval == 60
        assert detector.stage2_interval == 10
        assert detector.threshold == 7


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

    def test_single_high_intensity_frame(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=5, threshold=5)
        results = [(10.0, {"intensity": 7, "description": "action"})]
        segments = detector._merge_segments(results)
        assert len(segments) == 1
        assert segments[0].start_seconds == 10.0
        assert segments[0].end_seconds == 15.0
        assert segments[0].peak_intensity == 7

    def test_consecutive_frames_merge(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=5, threshold=5)
        results = [
            (10.0, {"intensity": 6, "description": "start"}),
            (15.0, {"intensity": 8, "description": "peak"}),
            (20.0, {"intensity": 7, "description": "end"}),
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 1
        assert segments[0].start_seconds == 10.0
        assert segments[0].end_seconds == 25.0
        assert segments[0].peak_intensity == 8

    def test_low_intensity_splits_segments(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=5, threshold=5)
        results = [
            (10.0, {"intensity": 7, "description": "first"}),
            (15.0, {"intensity": 2, "description": "low"}),
            (20.0, {"intensity": 2, "description": "low"}),
            (25.0, {"intensity": 2, "description": "low"}),
            (30.0, {"intensity": 8, "description": "second"}),
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 2

    def test_non_dict_result_treated_as_zero(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=5, threshold=5)
        results = [
            (10.0, {"intensity": 7, "description": "action"}),
            (15.0, "parse error"),
            (20.0, "parse error"),
            (25.0, "parse error"),
            (30.0, {"intensity": 6, "description": "more action"}),
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
            "intensity": 2,
            "reason": "lobby screen",
        }

        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30, threshold=5)
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
                    return {"scene": "battle", "intensity": 7, "reason": "intense fight"}
                return {"scene": "battle", "intensity": 3, "reason": "calm"}
            return {"intensity": 7, "description": "kills happening"}

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        detector = HighlightDetector(
            analyzer=analyzer, stage1_interval=30, stage2_interval=5, threshold=5
        )
        segments = detector.detect("/fake/video.mp4")

        assert len(segments) >= 1
        assert detector.stage1_summary["total_frames"] == 4
        assert detector.stage1_summary["candidate_frames"] == 1

    @patch("src.highlight_detector.extract_frames")
    def test_progress_callback_called(self, mock_extract: MagicMock) -> None:
        """Progress callback is invoked for each stage."""
        stage1_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3
        mock_extract.return_value = stage1_frames

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "scene": "lobby",
            "intensity": 2,
            "reason": "lobby screen",
        }

        progress_calls: list[tuple[int, int, int]] = []

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            progress_calls.append((phase, frames_done, frames_total))

        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30, threshold=5)
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
                    return {"scene": "battle", "intensity": 7, "reason": "intense fight"}
                return {"scene": "battle", "intensity": 3, "reason": "calm"}
            return {"intensity": 7, "description": "kills happening"}

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        progress_calls: list[tuple[int, int, int]] = []

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            progress_calls.append((phase, frames_done, frames_total))

        detector = HighlightDetector(
            analyzer=analyzer, stage1_interval=30, stage2_interval=5, threshold=5
        )
        detector.detect("/fake/video.mp4", progress_callback=on_progress)

        stage1_calls = [c for c in progress_calls if c[0] == 1]
        stage2_calls = [c for c in progress_calls if c[0] == 2]
        assert len(stage1_calls) == 4
        assert len(stage2_calls) == 3
        # Stage 2 final call should have frames_done == frames_total
        assert stage2_calls[-1][1] == stage2_calls[-1][2]

    @patch("src.highlight_detector.extract_frames")
    def test_battle_below_threshold_not_candidate(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 2

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "scene": "battle",
            "intensity": 3,
            "reason": "low action",
        }

        detector = HighlightDetector(analyzer=analyzer, stage1_interval=30, threshold=5)
        segments = detector.detect("/fake/video.mp4")

        assert segments == []
        assert detector.stage1_summary["battle_frames"] == 2
        assert detector.stage1_summary["candidate_frames"] == 0
