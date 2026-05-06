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

    def test_is_dead_halves_score(self) -> None:
        result = {"kills": 5, "assists": 2, "score_gain": 3, "special": 4, "is_dead": True}
        assert _compute_score(result) == (5 * 2 * 3 * 4) // 2

    def test_is_dead_false_no_penalty(self) -> None:
        result = {"kills": 5, "assists": 2, "score_gain": 3, "special": 4, "is_dead": False}
        assert _compute_score(result) == 5 * 2 * 3 * 4


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
        assert detector.interval == 5
        assert detector.threshold == 100

    def test_custom_params(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=10, threshold=200)
        assert detector.interval == 10
        assert detector.threshold == 200


class TestMergeSegments:
    """Tests for segment merging logic."""

    def test_empty_results(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        segments = detector._merge_segments([])
        assert segments == []

    def test_single_high_score_frame(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        # score = 5*3*3*3 = 135, above threshold 100
        results = [
            (
                10.0,
                {
                    "kills": 5,
                    "assists": 3,
                    "score_gain": 3,
                    "special": 3,
                    "description": "action",
                },
            )
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 1
        assert segments[0].start_seconds == 10.0
        assert segments[0].end_seconds == 15.0
        assert segments[0].peak_intensity == 135

    def test_consecutive_frames_merge(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        results = [
            (
                10.0,
                {
                    "kills": 5,
                    "assists": 3,
                    "score_gain": 3,
                    "special": 3,
                    "description": "start",
                },
            ),
            (
                15.0,
                {
                    "kills": 8,
                    "assists": 3,
                    "score_gain": 3,
                    "special": 3,
                    "description": "peak",
                },
            ),
            (
                20.0,
                {
                    "kills": 5,
                    "assists": 3,
                    "score_gain": 3,
                    "special": 3,
                    "description": "end",
                },
            ),
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 1
        assert segments[0].start_seconds == 10.0
        assert segments[0].end_seconds == 25.0
        assert segments[0].peak_intensity == 8 * 3 * 3 * 3

    def test_low_score_splits_segments(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        results = [
            (
                10.0,
                {
                    "kills": 5,
                    "assists": 3,
                    "score_gain": 3,
                    "special": 3,
                    "description": "first",
                },
            ),
            (
                15.0,
                {
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "description": "low",
                },
            ),
            (
                20.0,
                {
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "description": "low",
                },
            ),
            (
                25.0,
                {
                    "kills": 1,
                    "assists": 1,
                    "score_gain": 1,
                    "special": 1,
                    "description": "low",
                },
            ),
            (
                30.0,
                {
                    "kills": 8,
                    "assists": 3,
                    "score_gain": 3,
                    "special": 3,
                    "description": "second",
                },
            ),
        ]
        segments = detector._merge_segments(results)
        assert len(segments) == 2

    def test_non_dict_result_treated_as_zero(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        results = [
            (
                10.0,
                {
                    "kills": 5,
                    "assists": 3,
                    "score_gain": 3,
                    "special": 3,
                    "description": "action",
                },
            ),
            (15.0, "parse error"),
            (20.0, "parse error"),
            (25.0, "parse error"),
            (
                30.0,
                {
                    "kills": 5,
                    "assists": 3,
                    "score_gain": 3,
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
    def test_no_highlights_returns_empty(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "kills": 1,
            "assists": 1,
            "score_gain": 1,
            "special": 1,
            "description": "nothing happening",
        }

        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        segments = detector.detect("/fake/video.mp4")

        assert segments == []
        assert detector.scan_summary["total_frames"] == 3

    @patch("src.highlight_detector.extract_frames")
    def test_full_pipeline(self, mock_extract: MagicMock) -> None:
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 6
        mock_extract.return_value = frames

        def mock_analyze(frame, prompt, timestamp):
            if timestamp == "00m05s":
                return {
                    "kills": 5,
                    "assists": 3,
                    "score_gain": 3,
                    "special": 3,
                    "description": "intense fight",
                }
            return {
                "kills": 1,
                "assists": 1,
                "score_gain": 1,
                "special": 1,
                "description": "calm",
            }

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        segments = detector.detect("/fake/video.mp4")

        assert len(segments) == 1
        assert detector.scan_summary["total_frames"] == 6

    @patch("src.highlight_detector.extract_frames")
    def test_progress_callback_called(self, mock_extract: MagicMock) -> None:
        """Progress callback is invoked for each frame."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3
        mock_extract.return_value = frames

        analyzer = MagicMock()
        analyzer.concurrency = 1
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "kills": 1,
            "assists": 1,
            "score_gain": 1,
            "special": 1,
            "description": "nothing",
        }

        progress_calls: list[tuple[int, int, int]] = []

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            progress_calls.append((phase, frames_done, frames_total))

        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        detector.detect("/fake/video.mp4", progress_callback=on_progress)

        assert len(progress_calls) == 3
        assert all(c[0] == 1 for c in progress_calls)
        assert all(c[2] == 3 for c in progress_calls)

    @patch("src.highlight_detector.extract_frames")
    def test_parallel_execution(self, mock_extract: MagicMock) -> None:
        """Verify frames are analyzed in parallel via ThreadPoolExecutor."""
        import threading

        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        mock_extract.return_value = frames

        thread_ids: list[int] = []

        def mock_analyze(frame, prompt, timestamp):
            thread_ids.append(threading.current_thread().ident)
            return {
                "kills": 1,
                "assists": 1,
                "score_gain": 1,
                "special": 1,
                "description": "test",
            }

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        detector.detect("/fake/video.mp4")

        assert len(thread_ids) == 4

    @patch("src.highlight_detector.extract_frames")
    def test_all_low_score_not_highlighted(self, mock_extract: MagicMock) -> None:
        """All-1 scores don't pass threshold."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        mock_extract.return_value = frames

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "kills": 1,
            "assists": 1,
            "score_gain": 1,
            "special": 1,
            "description": "nothing happening",
        }

        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        segments = detector.detect("/fake/video.mp4")

        assert segments == []
        assert detector.scan_summary["total_frames"] == 4
