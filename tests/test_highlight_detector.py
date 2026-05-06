"""Tests for the highlight detection module."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.highlight_detector import (
    MAX_CLIP_SECONDS,
    WINDOW_SIZE,
    FrameAnalysis,
    HighlightDetector,
    HighlightSegment,
    _cap_segment_duration,
    _compute_frame_score,
    _compute_stage1_score,
)


class TestComputeFrameScore:
    """Tests for _compute_frame_score helper."""

    def test_all_defaults(self) -> None:
        result: dict = {}
        assert _compute_frame_score(result) == 1

    def test_kills_only(self) -> None:
        # kills=2 -> kills_score=7, rest=1 -> 7*1*1*1 = 7
        result = {"kills_in_log": 2}
        assert _compute_frame_score(result) == 7

    def test_kills_and_special(self) -> None:
        # kills=2 -> 7, special=true -> 10 => 7*1*1*10 = 70
        result = {"kills_in_log": 2, "my_special_active": True}
        assert _compute_frame_score(result) == 70

    def test_kills_3_caps_at_10(self) -> None:
        # kills=3 -> min(10, 1+9) = 10
        result = {"kills_in_log": 3}
        assert _compute_frame_score(result) == 10

    def test_kills_above_3_still_10(self) -> None:
        result = {"kills_in_log": 5}
        assert _compute_frame_score(result) == 10

    def test_assists(self) -> None:
        # assists=1 -> 4, kills=0 -> 1 => 1*4*1*1 = 4
        result = {"assists_in_log": 1}
        assert _compute_frame_score(result) == 4

    def test_team_score_increasing(self) -> None:
        # score_gain=5 when true
        result = {"team_score_increasing": True}
        assert _compute_frame_score(result) == 5

    def test_is_dead_halves_score(self) -> None:
        # kills=2 -> 7, is_dead -> 7//2 = 3
        result = {"kills_in_log": 2, "is_dead": True}
        assert _compute_frame_score(result) == 3

    def test_is_dead_minimum_1(self) -> None:
        # all defaults -> 1, is_dead -> 1//2 = 0 -> max(1, 0) = 1
        result = {"is_dead": True}
        assert _compute_frame_score(result) == 1

    def test_all_high(self) -> None:
        # kills=3->10, assists=3->10, score=true->5, special=true->10
        # 10*10*5*10 = 5000
        result = {
            "kills_in_log": 3,
            "assists_in_log": 3,
            "team_score_increasing": True,
            "my_special_active": True,
        }
        assert _compute_frame_score(result) == 5000

    def test_all_high_but_dead(self) -> None:
        result = {
            "kills_in_log": 3,
            "assists_in_log": 3,
            "team_score_increasing": True,
            "my_special_active": True,
            "is_dead": True,
        }
        assert _compute_frame_score(result) == 2500


class TestComputeStage1Score:
    """Tests for _compute_stage1_score helper."""

    def test_all_defaults(self) -> None:
        assert _compute_stage1_score({}) == 1

    def test_kills_only(self) -> None:
        result = {"kills_in_log": 2}
        # kills_score=7, special=1 -> 7
        assert _compute_stage1_score(result) == 7

    def test_special_active(self) -> None:
        result = {"my_special_active": True}
        # kills_score=1, special=10 -> 10
        assert _compute_stage1_score(result) == 10

    def test_is_dead_halves(self) -> None:
        result = {"kills_in_log": 1, "is_dead": True}
        # kills_score=4, special=1 -> 4//2 = 2
        assert _compute_stage1_score(result) == 2


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
        assert seg.frames == []

    def test_creation_with_frames(self) -> None:
        fa = FrameAnalysis(timestamp_seconds=10.0, kills_in_log=2, score=7)
        seg = HighlightSegment(
            start_seconds=10.0,
            end_seconds=25.0,
            peak_intensity=800,
            description="test",
            frames=[fa],
        )
        assert len(seg.frames) == 1
        assert seg.frames[0].kills_in_log == 2


class TestHighlightDetectorInit:
    """Tests for HighlightDetector initialization."""

    def test_defaults(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        assert detector.stage1_interval == 30
        assert detector.stage2_interval == 3
        assert detector.threshold == 100
        assert detector.max_highlights == 3

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


class TestFindBestWindows:
    """Tests for sliding window detection logic."""

    def test_empty_frames(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, threshold=10)
        segments = detector._find_best_windows([])
        assert segments == []

    def test_fewer_than_window_size_above_threshold(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=3, threshold=10)
        frames = [
            FrameAnalysis(timestamp_seconds=0.0, score=5),
            FrameAnalysis(timestamp_seconds=3.0, score=6),
        ]
        segments = detector._find_best_windows(frames)
        assert len(segments) == 1
        assert segments[0].peak_intensity == 11
        assert segments[0].start_seconds == 0.0
        assert segments[0].end_seconds == 6.0
        assert len(segments[0].frames) == 2

    def test_fewer_than_window_size_below_threshold(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=3, threshold=100)
        frames = [
            FrameAnalysis(timestamp_seconds=0.0, score=1),
            FrameAnalysis(timestamp_seconds=3.0, score=1),
        ]
        segments = detector._find_best_windows(frames)
        assert segments == []

    def test_best_window_selected(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(
            analyzer=analyzer, stage2_interval=3, threshold=10, max_highlights=1
        )
        # 10 frames, peak scores in frames 3-7
        frames = []
        for i in range(10):
            score = 50 if 3 <= i <= 7 else 1
            frames.append(FrameAnalysis(timestamp_seconds=i * 3.0, score=score))
        segments = detector._find_best_windows(frames)
        assert len(segments) == 1
        # Best window of 5 frames should be in the high-score region
        assert segments[0].peak_intensity == 250  # 5 * 50

    def test_overlapping_windows_excluded(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(
            analyzer=analyzer, stage2_interval=3, threshold=10, max_highlights=3
        )
        # 12 frames, all high score
        frames = [FrameAnalysis(timestamp_seconds=i * 3.0, score=20) for i in range(12)]
        segments = detector._find_best_windows(frames)
        # 12 frames, window_size=5 -> can fit at most 2 non-overlapping windows
        assert len(segments) == 2
        # Check they don't overlap
        for i in range(len(segments) - 1):
            assert segments[i].end_seconds <= segments[i + 1].start_seconds

    def test_max_highlights_limit(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(
            analyzer=analyzer, stage2_interval=3, threshold=10, max_highlights=1
        )
        frames = [FrameAnalysis(timestamp_seconds=i * 3.0, score=20) for i in range(15)]
        segments = detector._find_best_windows(frames)
        assert len(segments) == 1

    def test_threshold_filters_low_windows(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, stage2_interval=3, threshold=100)
        frames = [FrameAnalysis(timestamp_seconds=i * 3.0, score=10) for i in range(5)]
        # window score = 50, below threshold 100
        segments = detector._find_best_windows(frames)
        assert segments == []

    def test_frames_attached_to_segment(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(
            analyzer=analyzer, stage2_interval=3, threshold=10, max_highlights=1
        )
        frames = [
            FrameAnalysis(
                timestamp_seconds=i * 3.0,
                kills_in_log=i,
                score=20,
                description=f"frame {i}",
            )
            for i in range(5)
        ]
        segments = detector._find_best_windows(frames)
        assert len(segments) == 1
        assert len(segments[0].frames) == WINDOW_SIZE
        assert segments[0].frames[0].kills_in_log == 0
        assert segments[0].frames[4].kills_in_log == 4

    def test_sorted_by_time(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(
            analyzer=analyzer, stage2_interval=3, threshold=10, max_highlights=3
        )
        # 15 frames, two distinct peaks
        frames = []
        for i in range(15):
            if i < 5:
                score = 30
            elif i >= 10:
                score = 40
            else:
                score = 1
            frames.append(FrameAnalysis(timestamp_seconds=i * 3.0, score=score))
        segments = detector._find_best_windows(frames)
        assert len(segments) == 2
        # Should be sorted by start_seconds, not by score
        assert segments[0].start_seconds < segments[1].start_seconds


class TestDetectFlow:
    """Tests for the full detect() pipeline."""

    @patch("src.highlight_detector.extract_frames")
    def test_no_candidates_returns_empty(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "scene": "lobby",
            "kills_in_log": 0,
            "my_special_active": False,
            "is_dead": False,
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
                        "kills_in_log": 2,
                        "my_special_active": True,
                        "is_dead": False,
                        "reason": "intense fight",
                    }
                return {
                    "scene": "battle",
                    "kills_in_log": 0,
                    "my_special_active": False,
                    "is_dead": False,
                    "reason": "calm",
                }
            return {
                "kills_in_log": 2,
                "assists_in_log": 1,
                "team_score_increasing": True,
                "my_special_active": False,
                "is_dead": False,
                "description": "kills happening",
            }

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        detector = HighlightDetector(
            analyzer=analyzer, stage1_interval=30, stage2_interval=3, threshold=100
        )
        segments = detector.detect("/fake/video.mp4")

        # Stage 2: 3 frames, score=7*4*5*1=140 each, total=420 > threshold=100
        assert len(segments) >= 1
        assert detector.stage1_summary["total_frames"] == 4
        # max_highlights=3, so at most 3 candidates selected from 4 battle frames
        assert detector.stage1_summary["candidate_frames"] == 3

    @patch("src.highlight_detector.extract_frames")
    def test_progress_callback_called(self, mock_extract: MagicMock) -> None:
        """Progress callback is invoked for each stage."""
        stage1_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3
        mock_extract.return_value = stage1_frames

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.return_value = {
            "scene": "lobby",
            "kills_in_log": 0,
            "my_special_active": False,
            "is_dead": False,
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
                        "kills_in_log": 2,
                        "my_special_active": True,
                        "is_dead": False,
                        "reason": "intense fight",
                    }
                return {
                    "scene": "battle",
                    "kills_in_log": 0,
                    "my_special_active": False,
                    "is_dead": False,
                    "reason": "calm",
                }
            return {
                "kills_in_log": 2,
                "assists_in_log": 1,
                "team_score_increasing": True,
                "my_special_active": False,
                "is_dead": False,
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
        """Battle frames with low scores become candidates but don't pass threshold."""
        stage1_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 2
        stage2_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        mock_extract.side_effect = [stage1_frames, stage2_frames]

        def mock_analyze(frame, prompt, timestamp):
            from src.battle_analyzer import STAGE1_PROMPT

            if prompt == STAGE1_PROMPT:
                return {
                    "scene": "battle",
                    "kills_in_log": 0,
                    "my_special_active": False,
                    "is_dead": False,
                    "reason": "low action",
                }
            return {
                "kills_in_log": 0,
                "assists_in_log": 0,
                "team_score_increasing": False,
                "my_special_active": False,
                "is_dead": False,
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

    @patch("src.highlight_detector.extract_frames")
    def test_segments_have_frames(self, mock_extract: MagicMock) -> None:
        """Each highlight segment should contain FrameAnalysis details."""
        stage1_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 2
        # Need at least 5 stage2 frames for a full window
        stage2_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5

        mock_extract.side_effect = [stage1_frames, stage2_frames]

        def mock_analyze(frame, prompt, timestamp):
            from src.battle_analyzer import STAGE1_PROMPT

            if prompt == STAGE1_PROMPT:
                return {
                    "scene": "battle",
                    "kills_in_log": 2,
                    "my_special_active": False,
                    "is_dead": False,
                    "reason": "action",
                }
            return {
                "kills_in_log": 2,
                "assists_in_log": 0,
                "team_score_increasing": True,
                "my_special_active": False,
                "is_dead": False,
                "description": "kills happening",
            }

        analyzer = MagicMock()
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        detector = HighlightDetector(
            analyzer=analyzer, stage1_interval=30, stage2_interval=3, threshold=10
        )
        segments = detector.detect("/fake/video.mp4")

        assert len(segments) >= 1
        for seg in segments:
            assert len(seg.frames) > 0
            for fa in seg.frames:
                assert isinstance(fa, FrameAnalysis)
                assert fa.kills_in_log == 2
