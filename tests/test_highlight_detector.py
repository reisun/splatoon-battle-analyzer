"""Tests for the highlight detection module."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.highlight_detector import (
    HighlightDetector,
    HighlightSegment,
    ScoreBreakdown,
    _calc_score_count_gain,
    _compute_score,
    _isotonic_decreasing,
    _median_smooth,
    _normalize_counts,
    _ScoredFrame,
)
from src.scoring_config import ScoringConfig, ScoringWeights

_DEFAULT_CFG = ScoringConfig(
    weights=ScoringWeights(),
    death_penalty=0.5,
    score_count_gain_window_seconds=30,
)


class TestComputeScore:
    """Tests for _compute_score helper."""

    def test_all_ones(self) -> None:
        result = {"kills": 1, "score_count_gain": 1}
        b = _compute_score(result, _DEFAULT_CFG)
        # score_kills=1*1.0=1.0, score_count_gain=1+1*1.0=2.0, score=1.0*2.0=2.0
        assert b.score == 2.0
        assert b.score_kills == 1.0
        assert b.score_count_gain == 2.0
        assert b.score_dead == 0.0

    def test_all_max(self) -> None:
        result = {"kills": 4, "score_count_gain": 10}
        # score_kills=4.0, score_count_gain=1+10=11.0, score=4.0*11.0=44.0
        assert _compute_score(result, _DEFAULT_CFG).score == 44.0

    def test_mixed_values(self) -> None:
        result = {"kills": 3, "score_count_gain": 3}
        # score_kills=3.0, score_count_gain=1+3=4.0, score=3.0*4.0=12.0
        assert _compute_score(result, _DEFAULT_CFG).score == 12.0

    def test_missing_keys_default_to_zero(self) -> None:
        # score_kills=0.0, score_count_gain=1.0, score=0.0
        assert _compute_score({}, _DEFAULT_CFG).score == 0.0

    def test_kills_zero_is_valid(self) -> None:
        result = {"kills": 0, "score_count_gain": 1}
        # score_kills=0.0, score=0.0
        assert _compute_score(result, _DEFAULT_CFG).score == 0.0

    def test_clamps_above_four(self) -> None:
        result = {"kills": 99, "score_count_gain": 1}
        # kills clamped to 4, score_kills=4.0, score_count_gain=2.0, score=4.0*2.0=8.0
        assert _compute_score(result, _DEFAULT_CFG).score == 8.0

    def test_is_dead_returns_penalty_only(self) -> None:
        result = {"kills": 5, "score_count_gain": 3, "is_dead": True}
        b = _compute_score(result, _DEFAULT_CFG)
        assert b.score == 0.5
        assert b.score_kills == 0.0
        assert b.score_count_gain == 0.0
        assert b.score_dead == 0.5

    def test_is_dead_false_no_penalty(self) -> None:
        result = {"kills": 3, "score_count_gain": 3, "is_dead": False}
        # score_kills=3.0, score_count_gain=4.0, score=3.0*4.0=12.0
        assert _compute_score(result, _DEFAULT_CFG).score == 12.0

    def test_custom_weights(self) -> None:
        cfg = ScoringConfig(
            weights=ScoringWeights(kills=1.5, score_count_gain=0.5),
        )
        result = {"kills": 4, "score_count_gain": 3}
        # score_kills=4*1.5=6.0, score_count_gain=1+3*0.5=2.5, score=6.0*2.5=15.0
        assert _compute_score(result, cfg).score == 15.0

    def test_custom_death_penalty(self) -> None:
        cfg = ScoringConfig(death_penalty=3)
        result = {"kills": 10, "score_count_gain": 1, "is_dead": True}
        assert _compute_score(result, cfg).score == 3.0

    def test_enemy_score_gain_in_breakdown(self) -> None:
        """enemy_score_gain is passed through to breakdown."""
        result = {"kills": 1, "score_count_gain": 0, "enemy_score_gain": 5.0}
        b = _compute_score(result, _DEFAULT_CFG)
        assert b.enemy_score_gain == 5.0

    def test_score_count_gain_is_float(self) -> None:
        """score_count_gain in breakdown is a float, not int."""
        result = {"kills": 2, "score_count_gain": 3.5}
        b = _compute_score(result, _DEFAULT_CFG)
        assert isinstance(b.score_count_gain, float)
        # 1 + 3.5 * 1.0 = 4.5
        assert b.score_count_gain == 4.5
        # score = 2.0 * 4.5 = 9.0
        assert b.score == 9.0


class TestCalcScoreGain:
    """Tests for _calc_score_count_gain helper (forward-looking)."""

    def test_both_none(self) -> None:
        assert _calc_score_count_gain(None, None) == 0.0

    def test_cur_none(self) -> None:
        assert _calc_score_count_gain(None, 50) == 0.0

    def test_future_none(self) -> None:
        assert _calc_score_count_gain(50, None) == 0.0

    def test_no_change(self) -> None:
        assert _calc_score_count_gain(50, 50.0) == 0.0

    def test_future_lower(self) -> None:
        # cur=80, future_avg=60.0 -> 80-60.0 = 20.0
        assert _calc_score_count_gain(80, 60.0) == 20.0

    def test_future_much_lower(self) -> None:
        # cur=80, future_avg=0.0 -> 80-0.0 = 80.0
        assert _calc_score_count_gain(80, 0.0) == 80.0

    def test_large_diff(self) -> None:
        assert _calc_score_count_gain(100, 0.0) == 100.0

    def test_future_higher_clamps_to_zero(self) -> None:
        # cur=30, future_avg=50.0 -> 30-50.0 = -20.0 -> clamped to 0.0
        assert _calc_score_count_gain(30, 50.0) == 0.0

    def test_returns_float(self) -> None:
        result = _calc_score_count_gain(80, 60.0)
        assert isinstance(result, float)

    def test_fractional_future_avg(self) -> None:
        # cur=80, future_avg=65.5 -> 14.5
        assert _calc_score_count_gain(80, 65.5) == 14.5


class TestHighlightSegment:
    """Tests for HighlightSegment dataclass."""

    def test_creation(self) -> None:
        seg = HighlightSegment(
            start_seconds=10.0,
            end_seconds=25.0,
            peak_intensity=800,
        )
        assert seg.start_seconds == 10.0
        assert seg.end_seconds == 25.0
        assert seg.peak_intensity == 800


class TestHighlightDetectorInit:
    """Tests for HighlightDetector initialization."""

    def test_defaults(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        assert detector.interval == 5

    def test_custom_params(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=10)
        assert detector.interval == 10


class TestScoreFrames:
    """Tests for _score_frames (forward-looking score_count_gain)."""

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    def test_empty(self, _mock_cfg: MagicMock) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        assert detector._score_frames([]) == []

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    def test_computes_scores(self, _mock_cfg: MagicMock) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        results = [
            (0.0, {"kills": 4, "my_team_count": 80}),
            (5.0, {"kills": 4, "my_team_count": 80}),
            (10.0, {"kills": 4, "my_team_count": 80}),
            (15.0, {"kills": 4, "my_team_count": 60}),
            (20.0, {"kills": 4, "my_team_count": 60}),
            (25.0, {"kills": 4, "my_team_count": 60}),
        ]
        scored = detector._score_frames(results)
        assert len(scored) == 6
        # first frame: future avg of [80,80,60,60,60]=68.0 (float, no int rounding)
        # gain = 80-68.0 = 12.0
        # score_kills=4*1.0=4.0, score_count_gain=1+12.0*1.0=13.0, score=4.0*13.0=52.0
        assert scored[0].score == 52.0
        # last frame: no future -> gain=0
        # score_kills=4.0, score_count_gain=1.0, score=4.0*1.0=4.0
        assert scored[5].score == 4.0

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    def test_score_count_gain_uses_future_window(self, _mock_cfg: MagicMock) -> None:
        """score_count_gain は未来30秒分のカウント平均から計算."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        # 9 frames = 0s~40s. Window=30s/5s=6 frames.
        results = [
            (0.0, {"kills": 5, "special": True, "my_team_count": 100}),
            (5.0, {"kills": 1, "special": False, "my_team_count": 95}),
            (10.0, {"kills": 1, "special": False, "my_team_count": 90}),
            (15.0, {"kills": 1, "special": False, "my_team_count": 85}),
            (20.0, {"kills": 1, "special": False, "my_team_count": 80}),
            (25.0, {"kills": 1, "special": False, "my_team_count": 75}),
            (30.0, {"kills": 1, "special": False, "my_team_count": 70}),
            (35.0, {"kills": 1, "special": False, "my_team_count": 65}),
            (40.0, {"kills": 1, "special": False, "my_team_count": 60}),
        ]
        scored = detector._score_frames(results)
        # After median smoothing (radius=2), index 0 becomes 95.
        # Future 6 frames (smoothed) = [95,90,85,80,75,70], avg=82.5 (float!)
        # gain = 95-82.5 = 12.5
        assert scored[0].raw["score_count_gain"] == 12.5
        # At 40s (last): no future -> score_count_gain=0.0
        assert scored[8].raw["score_count_gain"] == 0.0

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    def test_enemy_score_gain_computed(self, _mock_cfg: MagicMock) -> None:
        """enemy_score_gain is computed from enemy_team_count."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        # Use enough frames so median smooth preserves the pattern.
        # enemy counts: 80,80,80,80,80,60,60,60,60,60
        results = [
            (float(i * 5), {"kills": 1, "my_team_count": 100, "enemy_team_count": 80})
            for i in range(5)
        ] + [
            (float((i + 5) * 5), {"kills": 1, "my_team_count": 100, "enemy_team_count": 60})
            for i in range(5)
        ]
        scored = detector._score_frames(results)
        # First frame enemy_score_gain should be > 0 (80 vs future avg < 80)
        assert scored[0].raw["enemy_score_gain"] > 0
        # Last frame: no future -> enemy_score_gain=0.0
        assert scored[9].raw["enemy_score_gain"] == 0.0
        # All frames have enemy_score_gain key
        for sf in scored:
            assert "enemy_score_gain" in sf.raw

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    def test_non_dict_result_scores_zero(self, _mock_cfg: MagicMock) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        results = [(0.0, "parse error")]
        scored = detector._score_frames(results)
        assert scored[0].score == 0

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    def test_sorts_by_timestamp(self, _mock_cfg: MagicMock) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        results = [
            (10.0, {"kills": 1, "score_count_gain": 1, "special": False}),
            (0.0, {"kills": 2, "score_count_gain": 1, "special": False}),
        ]
        scored = detector._score_frames(results)
        assert scored[0].timestamp == 0.0
        assert scored[1].timestamp == 10.0


class TestSelectWindows:
    """Tests for sliding window selection."""

    def test_empty(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        assert detector._select_windows([]) == []

    def test_low_scores_still_selected(self) -> None:
        """threshold廃止: 低スコアでもウィンドウは選出される."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        scored = [
            _ScoredFrame(0.0, 1.0, ScoreBreakdown(1.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(5.0, 1.0, ScoreBreakdown(1.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(10.0, 1.0, ScoreBreakdown(1.0, 0.0, 0.0, 0.0), {}),
        ]
        segments = detector._select_windows(scored)
        assert len(segments) == 1

    def test_single_window(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        scored = [
            _ScoredFrame(0.0, 10.0, ScoreBreakdown(10.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(5.0, 20.0, ScoreBreakdown(20.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(10.0, 10.0, ScoreBreakdown(10.0, 0.0, 0.0, 0.0), {}),
        ]
        segments = detector._select_windows(scored)
        assert len(segments) == 1
        assert segments[0].start_seconds == 0.0
        assert segments[0].end_seconds == 15.0
        assert segments[0].peak_intensity == 10 + 20 + 10

    def test_best_window_selected_by_sum(self) -> None:
        """Window with highest sum of scores is preferred."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        scored = [
            _ScoredFrame(0.0, 30.0, ScoreBreakdown(30.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(5.0, 1.0, ScoreBreakdown(1.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(10.0, 1.0, ScoreBreakdown(1.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(15.0, 20.0, ScoreBreakdown(20.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(20.0, 20.0, ScoreBreakdown(20.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(25.0, 20.0, ScoreBreakdown(20.0, 0.0, 0.0, 0.0), {}),
        ]
        segments = detector._select_windows(scored)
        assert len(segments) == 2
        sums = {s.start_seconds: s.peak_intensity for s in segments}
        assert sums[15.0] > sums[0.0]

    def test_non_overlapping(self) -> None:
        """選出されたウィンドウ同士は重複しない."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        scored = [
            _ScoredFrame(0.0, 20.0, ScoreBreakdown(20.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(5.0, 20.0, ScoreBreakdown(20.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(10.0, 20.0, ScoreBreakdown(20.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(15.0, 1.0, ScoreBreakdown(1.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(20.0, 1.0, ScoreBreakdown(1.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(25.0, 1.0, ScoreBreakdown(1.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(30.0, 15.0, ScoreBreakdown(15.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(35.0, 15.0, ScoreBreakdown(15.0, 0.0, 0.0, 0.0), {}),
            _ScoredFrame(40.0, 15.0, ScoreBreakdown(15.0, 0.0, 0.0, 0.0), {}),
        ]
        segments = detector._select_windows(scored)
        for i in range(len(segments) - 1):
            assert segments[i].end_seconds <= segments[i + 1].start_seconds
        starts = [s.start_seconds for s in segments]
        assert 0.0 in starts
        assert 30.0 in starts


class TestDetectFlow:
    """Tests for the full detect() pipeline."""

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=1)
    @patch("src.highlight_detector.extract_frames")
    def test_nawabari_uses_cv_detection(
        self,
        mock_extract: MagicMock,
        mock_kills: MagicMock,
        mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """3min (nawabari) uses CV-based kill/death detection."""
        mock_extract.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        analyzer = MagicMock()
        analyzer.concurrency = 4

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        segments = detector.detect("/fake/video.mp4", duration_type="3min")

        assert len(segments) >= 1
        assert detector.scan_summary["phase_a_frames"] == 3
        assert detector.scan_summary["phase_b_frames"] == 0
        assert mock_kills.call_count == 3
        assert mock_death.call_count == 3
        analyzer.analyze_frame_lower_only.assert_not_called()
        analyzer.analyze_frame_upper_only.assert_not_called()

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=2)
    @patch("src.highlight_detector.extract_frames")
    def test_ranked_uses_phase_a_and_b(
        self,
        mock_extract: MagicMock,
        mock_kills: MagicMock,
        mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """5min (ranked) uses Phase A upper + Phase B CV lower."""
        # Phase A at 15s: 4 frames (0, 15, 30, 45) for 60s video
        # Phase B at 5s: 12 frames (0, 5, 10, ..., 55)
        phase_a_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        phase_b_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 12

        # extract_frames called twice: first for 15s, then for 5s
        mock_extract.side_effect = [phase_a_frames, phase_b_frames]

        count_map_upper = {
            "00m00s": {"my_team_count": 100, "enemy_team_count": 100, "kills": 0, "is_dead": False},
            "00m15s": {"my_team_count": 80, "enemy_team_count": 80, "kills": 0, "is_dead": False},
            "00m30s": {"my_team_count": 60, "enemy_team_count": 60, "kills": 0, "is_dead": False},
            "00m45s": {"my_team_count": 40, "enemy_team_count": 40, "kills": 0, "is_dead": False},
        }

        def mock_upper(frame, timestamp):
            default = {
                "my_team_count": None,
                "enemy_team_count": None,
                "kills": 0,
                "is_dead": False,
            }
            return count_map_upper.get(timestamp, default)

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_upper_only.side_effect = mock_upper

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", duration_type="5min")

        assert detector.scan_summary["phase_a_frames"] == 4
        assert detector.scan_summary["phase_b_frames"] > 0
        assert detector.scan_summary["total_frames"] == 12
        analyzer.analyze_frame_upper_only.assert_called()
        analyzer.analyze_frame_lower_only.assert_not_called()
        assert mock_kills.call_count == 12
        assert mock_death.call_count == 12

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=0)
    @patch("src.highlight_detector.extract_frames")
    def test_ranked_phase_b_all_frames(
        self,
        mock_extract: MagicMock,
        mock_kills: MagicMock,
        mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """Phase B always analyzes all frames regardless of score_count_gain."""
        phase_a_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        phase_b_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 12
        mock_extract.side_effect = [phase_a_frames, phase_b_frames]

        # All same counts -> no gain, but Phase B still runs for all
        def mock_upper(frame, timestamp):
            return {"my_team_count": 100, "enemy_team_count": 100, "kills": 0, "is_dead": False}

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_upper_only.side_effect = mock_upper

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", duration_type="5min")

        assert detector.scan_summary["phase_b_frames"] == 12
        assert mock_kills.call_count == 12

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=1)
    @patch("src.highlight_detector.extract_frames")
    def test_progress_callback_nawabari(
        self,
        mock_extract: MagicMock,
        _mock_kills: MagicMock,
        _mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """Progress callback is invoked for nawabari (phase=1 only)."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3
        mock_extract.return_value = frames

        analyzer = MagicMock()
        analyzer.concurrency = 1

        progress_calls: list[tuple[int, int, int]] = []

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            progress_calls.append((phase, frames_done, frames_total))

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", duration_type="3min", progress_callback=on_progress)

        assert len(progress_calls) == 3
        assert all(c[0] == 1 for c in progress_calls)
        assert all(c[2] == 3 for c in progress_calls)

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=1)
    @patch("src.highlight_detector.extract_frames")
    def test_progress_callback_ranked(
        self,
        mock_extract: MagicMock,
        _mock_kills: MagicMock,
        _mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """Progress callback has phase=1 for A and phase=2 for B."""
        phase_a_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        phase_b_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 12
        mock_extract.side_effect = [phase_a_frames, phase_b_frames]

        count_map = {
            "00m00s": {"my_team_count": 100, "enemy_team_count": 100, "kills": 0, "is_dead": False},
            "00m15s": {"my_team_count": 80, "enemy_team_count": 80, "kills": 0, "is_dead": False},
            "00m30s": {"my_team_count": 60, "enemy_team_count": 60, "kills": 0, "is_dead": False},
            "00m45s": {"my_team_count": 40, "enemy_team_count": 40, "kills": 0, "is_dead": False},
        }

        def mock_upper(frame, timestamp):
            default = {
                "my_team_count": None,
                "enemy_team_count": None,
                "kills": 0,
                "is_dead": False,
            }
            return count_map.get(timestamp, default)

        analyzer = MagicMock()
        analyzer.concurrency = 1
        analyzer.analyze_frame_upper_only.side_effect = mock_upper

        progress_calls: list[tuple[int, int, int]] = []

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            progress_calls.append((phase, frames_done, frames_total))

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", duration_type="5min", progress_callback=on_progress)

        phase1_calls = [c for c in progress_calls if c[0] == 1]
        phase2_calls = [c for c in progress_calls if c[0] == 2]
        assert len(phase1_calls) == 4  # Phase A: 4 frames
        assert all(c[2] == 4 for c in phase1_calls)
        if phase2_calls:
            assert all(c[0] == 2 for c in phase2_calls)

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=1)
    @patch("src.highlight_detector.extract_frames")
    def test_enemy_score_gain_in_all_frames(
        self,
        mock_extract: MagicMock,
        _mock_kills: MagicMock,
        _mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """enemy_score_gain appears in all_frames output."""
        mock_extract.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        analyzer = MagicMock()
        analyzer.concurrency = 4

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", duration_type="3min")

        for frame in detector.all_frames:
            assert hasattr(frame, "enemy_score_gain")
            assert isinstance(frame.enemy_score_gain, float)


class TestMedianSmooth:
    """Tests for _median_smooth."""

    def test_no_outliers_unchanged(self) -> None:
        values = [100, 95, 100, 90, 100, 95, 100, 90, 100, 95]
        result = _median_smooth(values)
        assert all(90 <= v <= 100 for v in result)

    def test_isolated_misread_removed(self) -> None:
        values = [100, 100, 100, 12, 100, 100, 100, 100, 100, 100]
        result = _median_smooth(values)
        assert result[3] == 100

    def test_null_values_preserved(self) -> None:
        values = [100, None, 100, None, 100, 100, 100, 100, 100, 100]
        result = _median_smooth(values)
        assert result[1] is None
        assert result[3] is None
        assert result[0] == 100

    def test_sustained_decrease_preserved(self) -> None:
        values = [100, 100, 100, 100, 100, 50, 50, 50, 50, 50, 50, 50]
        result = _median_smooth(values)
        assert result[7] == 50
        assert result[8] == 50

    def test_small_dataset_unchanged(self) -> None:
        values = [100, 12]
        result = _median_smooth(values)
        assert len(result) == 2

    def test_real_data_pattern(self) -> None:
        """前半100、後半80付近、散発的な12/13."""
        values = [
            100,
            100,
            100,
            12,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            13,
            100,
            100,
            80,
            80,
            80,
            80,
            80,
            80,
        ]
        result = _median_smooth(values)
        assert result[3] == 100
        assert result[11] == 100
        assert result[16] == 80


class TestIsotonicDecreasing:
    """Tests for _isotonic_decreasing."""

    def test_already_decreasing(self) -> None:
        values = [100, 80, 60, 40, 20]
        result = _isotonic_decreasing(values)
        assert result == [100, 80, 60, 40, 20]

    def test_violation_merged(self) -> None:
        values = [100, 50, 80]
        result = _isotonic_decreasing(values)
        assert result[0] == 100
        # 50 and 80 violate monotone decrease, get merged
        assert result[1] == result[2]
        assert result[1] == round((50 + 80) / 2)

    def test_all_same(self) -> None:
        values = [100, 100, 100]
        result = _isotonic_decreasing(values)
        assert result == [100, 100, 100]

    def test_single_value(self) -> None:
        values = [50]
        result = _isotonic_decreasing(values)
        assert result == [50]

    def test_all_none(self) -> None:
        values: list[int | None] = [None, None]
        result = _isotonic_decreasing(values)
        assert result == [None, None]

    def test_none_values_skipped(self) -> None:
        values: list[int | None] = [100, None, 50, None, 20]
        result = _isotonic_decreasing(values)
        assert result[0] == 100
        assert result[1] is None
        assert result[2] == 50
        assert result[3] is None
        assert result[4] == 20

    def test_increasing_sequence_averaged(self) -> None:
        values = [10, 20, 30, 40, 50]
        result = _isotonic_decreasing(values)
        avg = round((10 + 20 + 30 + 40 + 50) / 5)
        assert all(v == avg for v in result)


class TestNormalizeCounts:
    """Tests for _normalize_counts."""

    def test_null_stays_null_when_no_previous(self) -> None:
        results = [(0.0, {"my_team_count": None, "enemy_team_count": None})]
        _normalize_counts(results)
        assert results[0][1]["my_team_count"] is None
        assert results[0][1]["my_team_count_raw"] is None

    def test_null_filled_with_previous(self) -> None:
        results = [
            (0.0, {"my_team_count": 80, "enemy_team_count": 90}),
            (5.0, {"my_team_count": None, "enemy_team_count": None}),
        ]
        _normalize_counts(results)
        assert results[1][1]["my_team_count"] == 80
        assert results[1][1]["enemy_team_count"] == 90
        assert results[1][1]["my_team_count_raw"] is None

    def test_first_value_very_low_normalized_to_100(self) -> None:
        results = [(0.0, {"my_team_count": 10, "enemy_team_count": 5})]
        _normalize_counts(results)
        assert results[0][1]["my_team_count"] == 100
        assert results[0][1]["my_team_count_raw"] == 10
        assert results[0][1]["enemy_team_count"] == 100
        assert results[0][1]["enemy_team_count_raw"] == 5

    def test_first_value_moderate_kept(self) -> None:
        results = [
            (0.0, {"my_team_count": 80, "enemy_team_count": 70}),
            (5.0, {"my_team_count": 80, "enemy_team_count": 70}),
            (10.0, {"my_team_count": 80, "enemy_team_count": 70}),
            (15.0, {"my_team_count": 80, "enemy_team_count": 70}),
            (20.0, {"my_team_count": 80, "enemy_team_count": 70}),
        ]
        _normalize_counts(results)
        assert results[0][1]["my_team_count"] == 80
        assert results[0][1]["enemy_team_count"] == 70

    def test_monotonic_decrease_enforced(self) -> None:
        """等張回帰により増加は許容されない."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (10.0, {"my_team_count": 80, "enemy_team_count": 80}),
            (15.0, {"my_team_count": 90, "enemy_team_count": 85}),
            (20.0, {"my_team_count": 80, "enemy_team_count": 80}),
        ]
        _normalize_counts(results)
        counts_my = [r["my_team_count"] for _, r in results if isinstance(r, dict)]
        counts_en = [r["enemy_team_count"] for _, r in results if isinstance(r, dict)]
        for i in range(1, len(counts_my)):
            assert counts_my[i] <= counts_my[i - 1], f"my_team not monotone at {i}"
            assert counts_en[i] <= counts_en[i - 1], f"enemy_team not monotone at {i}"

    def test_isolated_misread_corrected(self) -> None:
        """AIの孤立した誤読はメディアンフィルタで除去される."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (10.0, {"my_team_count": 12, "enemy_team_count": 0}),
            (15.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (20.0, {"my_team_count": 100, "enemy_team_count": 100}),
        ]
        _normalize_counts(results)
        assert results[2][1]["my_team_count"] == 100
        assert results[2][1]["enemy_team_count"] == 100
        assert results[2][1]["my_team_count_raw"] == 12

    def test_consistent_decrease_accepted(self) -> None:
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (10.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (15.0, {"my_team_count": 90, "enemy_team_count": 90}),
            (20.0, {"my_team_count": 80, "enemy_team_count": 80}),
            (25.0, {"my_team_count": 70, "enemy_team_count": 70}),
            (30.0, {"my_team_count": 60, "enemy_team_count": 60}),
            (35.0, {"my_team_count": 60, "enemy_team_count": 60}),
            (40.0, {"my_team_count": 60, "enemy_team_count": 60}),
        ]
        _normalize_counts(results)
        assert results[6][1]["my_team_count"] == 60
        assert results[6][1]["enemy_team_count"] == 60

    def test_knockout_push_accepted(self) -> None:
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (10.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (15.0, {"my_team_count": 95, "enemy_team_count": 95}),
            (20.0, {"my_team_count": 90, "enemy_team_count": 90}),
            (25.0, {"my_team_count": 80, "enemy_team_count": 80}),
            (30.0, {"my_team_count": 65, "enemy_team_count": 65}),
            (35.0, {"my_team_count": 45, "enemy_team_count": 45}),
            (40.0, {"my_team_count": 25, "enemy_team_count": 25}),
            (45.0, {"my_team_count": 10, "enemy_team_count": 10}),
            (50.0, {"my_team_count": 0, "enemy_team_count": 0}),
            (55.0, {"my_team_count": 0, "enemy_team_count": 0}),
            (60.0, {"my_team_count": 0, "enemy_team_count": 0}),
        ]
        _normalize_counts(results)
        my_counts = [r["my_team_count"] for _, r in results]
        for i in range(1, len(my_counts)):
            assert my_counts[i] <= my_counts[i - 1], f"not monotone at {i}"
        assert results[7][1]["my_team_count"] <= 50
        assert results[10][1]["my_team_count"] <= 10

    def test_raw_values_preserved(self) -> None:
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (10.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (15.0, {"my_team_count": 120, "enemy_team_count": 88}),
            (20.0, {"my_team_count": 95, "enemy_team_count": 92}),
            (25.0, {"my_team_count": 90, "enemy_team_count": 90}),
        ]
        _normalize_counts(results)
        assert results[3][1]["my_team_count_raw"] == 120
        assert results[4][1]["my_team_count_raw"] == 95

    def test_non_dict_results_skipped(self) -> None:
        results = [
            (0.0, "parse error"),
            (5.0, {"my_team_count": 80, "enemy_team_count": 80}),
        ]
        _normalize_counts(results)
        assert results[0][1] == "parse error"
        assert results[1][1]["my_team_count"] == 80

    def test_scattered_misreads_all_corrected(self) -> None:
        """散発的な誤読が全て除去される（実データパターン再現）."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (10.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (15.0, {"my_team_count": 12, "enemy_team_count": 100}),
            (20.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (25.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (30.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (35.0, {"my_team_count": 13, "enemy_team_count": 0}),
            (40.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (45.0, {"my_team_count": 100, "enemy_team_count": 100}),
        ]
        _normalize_counts(results)
        for i in range(10):
            assert results[i][1]["my_team_count"] == 100, f"frame {i}"
            assert results[i][1]["enemy_team_count"] == 100, f"frame {i}"

    def test_real_data_single_misread_no_cascade(self) -> None:
        """1つの誤読値が全体をカスケード破壊しないことを確認."""
        results = [
            (float(i * 5), {"my_team_count": 100, "enemy_team_count": 100}) for i in range(10)
        ]
        # 1つだけ低い誤読を挿入
        results[5] = (25.0, {"my_team_count": 2, "enemy_team_count": 0})
        _normalize_counts(results)
        # 誤読後のフレームが誤読値に固定されないこと
        assert results[6][1]["my_team_count"] >= 50
        assert results[7][1]["my_team_count"] >= 50

    def test_steep_descent_tracked(self) -> None:
        """エリアKOの急降下(100→2)が正規化後も追従される."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (10.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (15.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (20.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (25.0, {"my_team_count": 70, "enemy_team_count": 47}),
            (30.0, {"my_team_count": 45, "enemy_team_count": 52}),
            (35.0, {"my_team_count": 35, "enemy_team_count": 45}),
            (40.0, {"my_team_count": 27, "enemy_team_count": 45}),
            (45.0, {"my_team_count": 18, "enemy_team_count": 45}),
            (50.0, {"my_team_count": 10, "enemy_team_count": 45}),
            (55.0, {"my_team_count": 2, "enemy_team_count": 45}),
            (60.0, {"my_team_count": None, "enemy_team_count": None}),
            (65.0, {"my_team_count": None, "enemy_team_count": None}),
        ]
        _normalize_counts(results)
        assert results[11][1]["my_team_count"] <= 10
        assert results[7][1]["my_team_count"] <= 40
        my_counts = [
            r["my_team_count"]
            for _, r in results
            if isinstance(r, dict) and r["my_team_count"] is not None
        ]
        for i in range(1, len(my_counts)):
            assert my_counts[i] <= my_counts[i - 1], f"not monotone at {i}"


class TestCountRailSwap:
    """Tests for count rail detection and my/enemy count swapping."""

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=1)
    @patch("src.highlight_detector.extract_frames")
    def test_rail_majority_triggers_swap(
        self,
        mock_extract: MagicMock,
        _mock_kills: MagicMock,
        _mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """When has_count_rail=True is majority, counts are swapped."""
        phase_a_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        phase_b_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 12
        mock_extract.side_effect = [phase_a_frames, phase_b_frames]

        # 3/4 frames have has_count_rail=True -> majority -> swap
        upper_results = [
            {"my_team_count": 100, "enemy_team_count": 80, "has_count_rail": True},
            {"my_team_count": 90, "enemy_team_count": 70, "has_count_rail": True},
            {"my_team_count": 80, "enemy_team_count": 60, "has_count_rail": True},
            {"my_team_count": 70, "enemy_team_count": 50, "has_count_rail": False},
        ]
        call_count = [0]

        def mock_upper(frame, timestamp):
            idx = call_count[0]
            call_count[0] += 1
            return upper_results[min(idx, len(upper_results) - 1)]

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_upper_only.side_effect = mock_upper

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", duration_type="5min")

        assert detector.scan_summary["count_swapped"] is True

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=1)
    @patch("src.highlight_detector.extract_frames")
    def test_rail_minority_no_swap(
        self,
        mock_extract: MagicMock,
        _mock_kills: MagicMock,
        _mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """When has_count_rail=True is minority, no swap occurs."""
        phase_a_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 4
        phase_b_frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 12
        mock_extract.side_effect = [phase_a_frames, phase_b_frames]

        # 1/4 frames have has_count_rail=True -> minority -> no swap
        upper_results = [
            {"my_team_count": 100, "enemy_team_count": 80, "has_count_rail": True},
            {"my_team_count": 90, "enemy_team_count": 70, "has_count_rail": False},
            {"my_team_count": 80, "enemy_team_count": 60, "has_count_rail": False},
            {"my_team_count": 70, "enemy_team_count": 50, "has_count_rail": False},
        ]
        call_count = [0]

        def mock_upper(frame, timestamp):
            idx = call_count[0]
            call_count[0] += 1
            return upper_results[min(idx, len(upper_results) - 1)]

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_upper_only.side_effect = mock_upper

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", duration_type="5min")

        assert detector.scan_summary["count_swapped"] is False

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.detect_death", return_value=False)
    @patch("src.highlight_detector.detect_kills", return_value=1)
    @patch("src.highlight_detector.extract_frames")
    def test_nawabari_count_swapped_false(
        self,
        mock_extract: MagicMock,
        _mock_kills: MagicMock,
        _mock_death: MagicMock,
        _mc: MagicMock,
    ) -> None:
        """Nawabari mode always has count_swapped=False."""
        mock_extract.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3

        analyzer = MagicMock()
        analyzer.concurrency = 4

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", duration_type="3min")

        assert detector.scan_summary["count_swapped"] is False
