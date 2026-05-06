"""Tests for the highlight detection module."""

import math
from unittest.mock import MagicMock, patch

import numpy as np

from src.highlight_detector import (
    HighlightDetector,
    HighlightSegment,
    _calc_score_gain,
    _compute_iqr_fence,
    _compute_score,
    _normalize_counts,
    _ScoredFrame,
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
        result = {
            "kills": 5,
            "assists": 2,
            "score_gain": 3,
            "special": 4,
            "is_dead": True,
        }
        assert _compute_score(result) == (5 * 2 * 3 * 4) // 2

    def test_is_dead_false_no_penalty(self) -> None:
        result = {
            "kills": 5,
            "assists": 2,
            "score_gain": 3,
            "special": 4,
            "is_dead": False,
        }
        assert _compute_score(result) == 5 * 2 * 3 * 4


class TestCalcScoreGain:
    """Tests for _calc_score_gain helper."""

    def test_both_none(self) -> None:
        assert _calc_score_gain(None, None) == 1

    def test_prev_none(self) -> None:
        assert _calc_score_gain(None, 50) == 1

    def test_cur_none(self) -> None:
        assert _calc_score_gain(50, None) == 1

    def test_no_change(self) -> None:
        assert _calc_score_gain(50, 50) == 1

    def test_count_decreased(self) -> None:
        assert _calc_score_gain(50, 40) == 2

    def test_count_decreased_large(self) -> None:
        assert _calc_score_gain(80, 0) == 9

    def test_clamps_to_ten(self) -> None:
        assert _calc_score_gain(100, 0) == 10

    def test_count_increased_clamps_to_one(self) -> None:
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
        assert detector.threshold == 100

    def test_custom_params(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=10, threshold=200)
        assert detector.interval == 10
        assert detector.threshold == 200


class TestScoreFrames:
    """Tests for _score_frames."""

    def test_empty(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        assert detector._score_frames([]) == []

    def test_computes_scores(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        results = [
            (0.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 80}),
            (5.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 60}),
        ]
        scored = detector._score_frames(results)
        assert len(scored) == 2
        # first frame: prev_count=None, score_gain=1 -> 5*3*1*3=45
        assert scored[0].score == 45
        # second frame: (80-60)/10+1=3, score_gain=3 -> 5*3*3*3=135
        assert scored[1].score == 135

    def test_score_gain_uses_window_average(self) -> None:
        """score_gain の基準値は直近40秒分の平均カウント."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5)
        # 9 frames = 0s~40s. Window=40s/5s=8 frames max.
        # Counts decrease by 5 each, which is consistent, so normalization passes.
        results = [
            (0.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 100}),
            (5.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 95}),
            (10.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 90}),
            (15.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 85}),
            (20.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 80}),
            (25.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 75}),
            (30.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 70}),
            (35.0, {"kills": 1, "assists": 1, "special": 1, "my_team_count": 65}),
            (40.0, {"kills": 5, "assists": 3, "special": 3, "my_team_count": 60}),
        ]
        scored = detector._score_frames(results)
        # At 40s: window has frames 0-35s (8 frames): 100,95,90,85,80,75,70,65
        # avg = 82.5 -> int = 82. cur_count = 60 (normal decrease of 5).
        # gain = (82-60)/10+1 = 3.2 -> int = 3
        assert scored[8].raw["score_gain"] == 3

    def test_non_dict_result_scores_zero(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer)
        results = [(0.0, "parse error")]
        scored = detector._score_frames(results)
        assert scored[0].score == 0

    def test_sorts_by_timestamp(self) -> None:
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

    def test_all_below_threshold(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        scored = [
            _ScoredFrame(0.0, 1, "", {}),
            _ScoredFrame(5.0, 1, "", {}),
            _ScoredFrame(10.0, 1, "", {}),
        ]
        assert detector._select_windows(scored) == []

    def test_single_window(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        scored = [
            _ScoredFrame(0.0, 135, "a", {}),
            _ScoredFrame(5.0, 216, "b", {}),
            _ScoredFrame(10.0, 135, "c", {}),
        ]
        segments = detector._select_windows(scored)
        assert len(segments) == 1
        assert segments[0].start_seconds == 0.0
        assert segments[0].end_seconds == 15.0
        assert segments[0].peak_intensity == 135 + 216 + 135

    def test_best_window_selected_by_sum(self) -> None:
        """Window with highest sum of scores is preferred over peak."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        scored = [
            _ScoredFrame(0.0, 500, "spike", {}),
            _ScoredFrame(5.0, 1, "", {}),
            _ScoredFrame(10.0, 1, "", {}),
            _ScoredFrame(15.0, 200, "a", {}),
            _ScoredFrame(20.0, 200, "b", {}),
            _ScoredFrame(25.0, 200, "c", {}),
        ]
        segments = detector._select_windows(scored)
        assert len(segments) == 2
        sums = {s.start_seconds: s.peak_intensity for s in segments}
        assert sums[15.0] > sums[0.0]

    def test_non_overlapping(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        scored = [
            _ScoredFrame(0.0, 200, "a", {}),
            _ScoredFrame(5.0, 200, "b", {}),
            _ScoredFrame(10.0, 200, "c", {}),
            _ScoredFrame(15.0, 1, "", {}),
            _ScoredFrame(20.0, 1, "", {}),
            _ScoredFrame(25.0, 1, "", {}),
            _ScoredFrame(30.0, 150, "d", {}),
            _ScoredFrame(35.0, 150, "e", {}),
            _ScoredFrame(40.0, 150, "f", {}),
        ]
        segments = detector._select_windows(scored)
        assert len(segments) == 2
        assert segments[0].start_seconds == 0.0
        assert segments[1].start_seconds == 30.0

    def test_single_high_frame_fallback(self) -> None:
        """When no full window has a high-score frame, single frames work."""
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        scored = [_ScoredFrame(10.0, 200, "action", {})]
        segments = detector._select_windows(scored)
        assert len(segments) == 1
        assert segments[0].start_seconds == 10.0

    def test_descriptions_summarized(self) -> None:
        analyzer = MagicMock()
        detector = HighlightDetector(analyzer=analyzer, interval=5, threshold=100)
        scored = [
            _ScoredFrame(0.0, 200, "first", {}),
            _ScoredFrame(5.0, 200, "second", {}),
            _ScoredFrame(10.0, 200, "third", {}),
        ]
        segments = detector._select_windows(scored)
        assert "first" in segments[0].description
        assert "second" in segments[0].description
        assert "third" in segments[0].description


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


class TestComputeIqrFence:
    """Tests for _compute_iqr_fence."""

    def test_insufficient_data_returns_inf(self) -> None:
        assert _compute_iqr_fence([1, 2, 3]) == math.inf

    def test_uniform_data(self) -> None:
        # All same -> Q1=Q3=5, IQR=0, fence=5
        assert _compute_iqr_fence([5, 5, 5, 5]) == 5.0

    def test_spread_data(self) -> None:
        # [2, 4, 6, 8] sorted. n=4, Q1=sorted[1]=4, Q3=sorted[3]=8
        # IQR=4, fence=8+1.5*4=14
        assert _compute_iqr_fence([2, 4, 6, 8]) == 14.0

    def test_larger_dataset(self) -> None:
        # [1,2,3,4,5,6,7,8] n=8, Q1=sorted[2]=3, Q3=sorted[6]=7
        # IQR=4, fence=7+6=13
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        assert _compute_iqr_fence(data) == 13.0


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
        results = [(0.0, {"my_team_count": 80, "enemy_team_count": 70})]
        _normalize_counts(results)
        assert results[0][1]["my_team_count"] == 80
        assert results[0][1]["enemy_team_count"] == 70

    def test_increasing_value_treated_as_anomaly(self) -> None:
        results = [
            (0.0, {"my_team_count": 80, "enemy_team_count": 80}),
            (5.0, {"my_team_count": 90, "enemy_team_count": 85}),
        ]
        _normalize_counts(results)
        assert results[1][1]["my_team_count"] == 80
        assert results[1][1]["enemy_team_count"] == 80
        assert results[1][1]["my_team_count_raw"] == 90

    def test_isolated_spike_detected(self) -> None:
        """安定した推移の中で孤立した急落は異常値として検出される."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 95, "enemy_team_count": 95}),
            (10.0, {"my_team_count": 90, "enemy_team_count": 90}),
            (15.0, {"my_team_count": 85, "enemy_team_count": 85}),
            (20.0, {"my_team_count": 80, "enemy_team_count": 80}),
            (25.0, {"my_team_count": 75, "enemy_team_count": 75}),
            (30.0, {"my_team_count": 70, "enemy_team_count": 70}),
            # 孤立した急落（単発）: deltas=[5,5,5,5,5,5] → IQR=0, fence=5
            # delta=40 > 5 → フェンス超え & 非連続 → 異常値
            (35.0, {"my_team_count": 30, "enemy_team_count": 30}),
            (40.0, {"my_team_count": 65, "enemy_team_count": 65}),
        ]
        _normalize_counts(results)
        assert results[7][1]["my_team_count"] == 70
        assert results[7][1]["my_team_count_raw"] == 30

    def test_consistent_decrease_accepted(self) -> None:
        """一貫した減少ペースなら正常."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 90, "enemy_team_count": 90}),
            (10.0, {"my_team_count": 80, "enemy_team_count": 80}),
            (15.0, {"my_team_count": 70, "enemy_team_count": 70}),
            (20.0, {"my_team_count": 60, "enemy_team_count": 60}),
        ]
        _normalize_counts(results)
        assert results[4][1]["my_team_count"] == 60
        assert results[4][1]["enemy_team_count"] == 60

    def test_knockout_push_accepted(self) -> None:
        """連続した大幅減少（ノックアウト推進）は正常と判定される."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 95, "enemy_team_count": 95}),
            (10.0, {"my_team_count": 90, "enemy_team_count": 90}),
            (15.0, {"my_team_count": 85, "enemy_team_count": 85}),
            # ノックアウト推進: 連続して大幅な減少
            (20.0, {"my_team_count": 55, "enemy_team_count": 55}),
            (25.0, {"my_team_count": 25, "enemy_team_count": 25}),
            (30.0, {"my_team_count": 0, "enemy_team_count": 0}),
        ]
        _normalize_counts(results)
        # 連続(3フレーム)しているため正常扱い
        assert results[4][1]["my_team_count"] == 55
        assert results[5][1]["my_team_count"] == 25
        assert results[6][1]["my_team_count"] == 0

    def test_raw_values_preserved(self) -> None:
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 95, "enemy_team_count": 92}),
            (10.0, {"my_team_count": 120, "enemy_team_count": 88}),
        ]
        _normalize_counts(results)
        assert results[0][1]["my_team_count_raw"] == 100
        assert results[1][1]["my_team_count_raw"] == 95
        assert results[2][1]["my_team_count_raw"] == 120
        assert results[2][1]["my_team_count"] == 95

    def test_non_dict_results_skipped(self) -> None:
        results = [
            (0.0, "parse error"),
            (5.0, {"my_team_count": 80, "enemy_team_count": 80}),
        ]
        _normalize_counts(results)
        assert results[0][1] == "parse error"
        assert results[1][1]["my_team_count"] == 80

    def test_few_data_points_all_accepted(self) -> None:
        """データが少ない場合はIQR算出不能のため全て正常扱い."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 60, "enemy_team_count": 60}),
        ]
        _normalize_counts(results)
        # fence=inf → 全減少が正常
        assert results[1][1]["my_team_count"] == 60

    def test_isolated_anomaly_does_not_affect_neighbors(self) -> None:
        """孤立した異常値は弾かれ、隣接する正常値はそのまま."""
        results = [
            (0.0, {"my_team_count": 100, "enemy_team_count": 100}),
            (5.0, {"my_team_count": 95, "enemy_team_count": 95}),
            (10.0, {"my_team_count": 90, "enemy_team_count": 90}),
            (15.0, {"my_team_count": 85, "enemy_team_count": 85}),
            (20.0, {"my_team_count": 80, "enemy_team_count": 80}),
            (25.0, {"my_team_count": 75, "enemy_team_count": 75}),
            (30.0, {"my_team_count": 70, "enemy_team_count": 70}),
            # 孤立した急落
            (35.0, {"my_team_count": 30, "enemy_team_count": 30}),
            # 正常な減少に戻る
            (40.0, {"my_team_count": 65, "enemy_team_count": 65}),
        ]
        _normalize_counts(results)
        assert results[7][1]["my_team_count"] == 70
        assert results[8][1]["my_team_count"] == 65
