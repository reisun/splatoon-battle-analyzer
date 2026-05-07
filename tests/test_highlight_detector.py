"""Tests for the highlight detection module."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.highlight_detector import (
    HighlightDetector,
    HighlightSegment,
    _calc_score_gain,
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
    score_gain_window_seconds=30,
)


class TestComputeScore:
    """Tests for _compute_score helper."""

    def test_all_ones(self) -> None:
        result = {"kills": 1, "assists": 1, "score_gain": 1, "special": 1}
        assert _compute_score(result, _DEFAULT_CFG) == 4

    def test_all_tens(self) -> None:
        result = {"kills": 10, "assists": 10, "score_gain": 10, "special": 10}
        assert _compute_score(result, _DEFAULT_CFG) == 40

    def test_mixed_values(self) -> None:
        result = {"kills": 5, "assists": 2, "score_gain": 3, "special": 4}
        assert _compute_score(result, _DEFAULT_CFG) == 5 + 2 + 3 + 4

    def test_missing_keys_default_to_one(self) -> None:
        assert _compute_score({}, _DEFAULT_CFG) == 4

    def test_clamps_below_one(self) -> None:
        result = {"kills": 0, "assists": -5, "score_gain": 1, "special": 1}
        assert _compute_score(result, _DEFAULT_CFG) == 4

    def test_clamps_above_ten(self) -> None:
        result = {"kills": 99, "assists": 1, "score_gain": 1, "special": 1}
        assert _compute_score(result, _DEFAULT_CFG) == 13

    def test_is_dead_returns_penalty_only(self) -> None:
        result = {
            "kills": 5,
            "assists": 2,
            "score_gain": 3,
            "special": 4,
            "is_dead": True,
        }
        assert _compute_score(result, _DEFAULT_CFG) == int(0.5)

    def test_is_dead_false_no_penalty(self) -> None:
        result = {
            "kills": 5,
            "assists": 2,
            "score_gain": 3,
            "special": 4,
            "is_dead": False,
        }
        assert _compute_score(result, _DEFAULT_CFG) == 5 + 2 + 3 + 4

    def test_custom_weights(self) -> None:
        cfg = ScoringConfig(
            weights=ScoringWeights(kills=1.5, assists=1.0, score_gain=1.0, special=1.0),
        )
        result = {"kills": 4, "assists": 2, "score_gain": 3, "special": 2}
        assert _compute_score(result, cfg) == int(4 * 1.5 + 2 + 3 + 2)

    def test_custom_death_penalty(self) -> None:
        cfg = ScoringConfig(death_penalty=3)
        result = {"kills": 10, "assists": 1, "score_gain": 1, "special": 1, "is_dead": True}
        assert _compute_score(result, cfg) == 3


class TestCalcScoreGain:
    """Tests for _calc_score_gain helper (forward-looking)."""

    def test_both_none(self) -> None:
        assert _calc_score_gain(None, None) == 1

    def test_cur_none(self) -> None:
        assert _calc_score_gain(None, 50) == 1

    def test_future_none(self) -> None:
        assert _calc_score_gain(50, None) == 1

    def test_no_change(self) -> None:
        assert _calc_score_gain(50, 50) == 1

    def test_future_lower(self) -> None:
        # cur=80, future_avg=60 -> (80-60)/10+1 = 3
        assert _calc_score_gain(80, 60) == 3

    def test_future_much_lower(self) -> None:
        # cur=80, future_avg=0 -> (80-0)/10+1 = 9
        assert _calc_score_gain(80, 0) == 9

    def test_clamps_to_ten(self) -> None:
        assert _calc_score_gain(100, 0) == 10

    def test_future_higher_clamps_to_one(self) -> None:
        # cur=30, future_avg=50 -> (30-50)/10+1 = -1 -> clamped to 1
        assert _calc_score_gain(30, 50) == 1


class TestHighlightSegment:
    """Tests for HighlightSegment dataclass."""

    def test_creation(self) -> None:
        seg = HighlightSegment(
            start_seconds=10.0,
            end_seconds=25.0,
            peak_intensity=800,
            description="test",
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

    def test_custom_params(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=10)
        assert detector.interval == 10


class TestScoreFrames:
    """Tests for _score_frames (forward-looking score_gain)."""

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
            (0.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 80}),
            (5.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 80}),
            (10.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 80}),
            (15.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 60}),
            (20.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 60}),
            (25.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 60}),
        ]
        scored = detector._score_frames(results)
        assert len(scored) == 6
        # first frame: future window covers 60s, avg of 80,80,60,60,60=68
        # gain = (80-68)/10+1 = 2.2 -> int=2 -> 5+3+2+3=13
        assert scored[0].score == 13
        # last frame: no future -> score_gain=1 -> 5+3+1+3=12
        assert scored[5].score == 12

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    def test_score_gain_uses_future_window(self, _mock_cfg: MagicMock) -> None:
        """score_gain は未来30秒分のカウント平均から計算."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        # 9 frames = 0s~40s. Window=30s/5s=6 frames.
        results = [
            (0.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 100}),
            (5.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 95}),
            (10.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 90}),
            (15.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 85}),
            (20.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 80}),
            (25.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 75}),
            (30.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 70}),
            (35.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 65}),
            (40.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 60}),
        ]
        scored = detector._score_frames(results)
        # At 0s: future 6 frames = [95,90,85,80,75,70], avg=82.5->82
        # gain = (100-82)/10+1 = 2.8 -> int=2
        assert scored[0].raw["score_gain"] == 2
        # At 40s (last): no future -> score_gain=1
        assert scored[8].raw["score_gain"] == 1

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
            (10.0, {"kills": 1, "assists": 1, "score_gain": 1, "special": 1}),
            (0.0, {"kills": 2, "assists": 1, "score_gain": 1, "special": 1}),
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
            _ScoredFrame(0.0, 1, "a", {}),
            _ScoredFrame(5.0, 1, "b", {}),
            _ScoredFrame(10.0, 1, "c", {}),
        ]
        segments = detector._select_windows(scored)
        assert len(segments) == 1

    def test_single_window(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        scored = [
            _ScoredFrame(0.0, 10, "a", {}),
            _ScoredFrame(5.0, 20, "b", {}),
            _ScoredFrame(10.0, 10, "c", {}),
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
            _ScoredFrame(0.0, 30, "spike", {}),
            _ScoredFrame(5.0, 1, "", {}),
            _ScoredFrame(10.0, 1, "", {}),
            _ScoredFrame(15.0, 20, "a", {}),
            _ScoredFrame(20.0, 20, "b", {}),
            _ScoredFrame(25.0, 20, "c", {}),
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
            _ScoredFrame(0.0, 20, "a", {}),
            _ScoredFrame(5.0, 20, "b", {}),
            _ScoredFrame(10.0, 20, "c", {}),
            _ScoredFrame(15.0, 1, "", {}),
            _ScoredFrame(20.0, 1, "", {}),
            _ScoredFrame(25.0, 1, "", {}),
            _ScoredFrame(30.0, 15, "d", {}),
            _ScoredFrame(35.0, 15, "e", {}),
            _ScoredFrame(40.0, 15, "f", {}),
        ]
        segments = detector._select_windows(scored)
        for i in range(len(segments) - 1):
            assert segments[i].end_seconds <= segments[i + 1].start_seconds
        starts = [s.start_seconds for s in segments]
        assert 0.0 in starts
        assert 30.0 in starts

    def test_descriptions_summarized(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        scored = [
            _ScoredFrame(0.0, 20, "first", {}),
            _ScoredFrame(5.0, 20, "second", {}),
            _ScoredFrame(10.0, 20, "third", {}),
        ]
        segments = detector._select_windows(scored)
        assert "first" in segments[0].description
        assert "second" in segments[0].description
        assert "third" in segments[0].description


class TestDetectFlow:
    """Tests for the full detect() pipeline."""

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.extract_frames")
    def test_low_scores_still_produce_segments(
        self, mock_extract: MagicMock, _mc: MagicMock
    ) -> None:
        """threshold廃止: 低スコアでも常にセグメントが返る."""
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

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        segments = detector.detect("/fake/video.mp4")

        assert len(segments) >= 1
        assert detector.scan_summary["total_frames"] == 3

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.extract_frames")
    def test_full_pipeline(self, mock_extract: MagicMock, _mc: MagicMock) -> None:
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 6
        mock_extract.return_value = frames

        count_map = {
            "00m00s": 100,
            "00m05s": 80,
            "00m10s": 60,
            "00m15s": 40,
            "00m20s": 40,
            "00m25s": 40,
        }

        def mock_analyze(frame, prompt, timestamp):
            count = count_map.get(timestamp, 100)
            if timestamp in ("00m05s", "00m10s", "00m15s"):
                return {
                    "kills": 5,
                    "assists": 3,
                    "special": 3,
                    "my_team_count": count,
                    "description": "intense fight",
                }
            return {
                "kills": 1,
                "assists": 1,
                "special": 1,
                "my_team_count": count,
                "description": "calm",
            }

        analyzer = MagicMock()
        analyzer.concurrency = 4
        analyzer.analyze_frame_from_memory_with_prompt.side_effect = mock_analyze

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        segments = detector.detect("/fake/video.mp4")

        assert len(segments) >= 1
        assert detector.scan_summary["total_frames"] == 6

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.extract_frames")
    def test_progress_callback_called(self, mock_extract: MagicMock, _mc: MagicMock) -> None:
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

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4", progress_callback=on_progress)

        assert len(progress_calls) == 3
        assert all(c[0] == 1 for c in progress_calls)
        assert all(c[2] == 3 for c in progress_calls)

    @patch("src.highlight_detector.load_scoring_config", return_value=_DEFAULT_CFG)
    @patch("src.highlight_detector.extract_frames")
    def test_parallel_execution(self, mock_extract: MagicMock, _mc: MagicMock) -> None:
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

        detector = HighlightDetector(analyzer=analyzer, interval=5)
        detector.detect("/fake/video.mp4")

        assert len(thread_ids) == 4


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
