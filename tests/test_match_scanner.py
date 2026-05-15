"""Tests for match boundary scanner."""

from src.match_scanner import (
    TimerReading,
    calc_match_start,
    cluster_readings,
    determine_rule,
    parse_timer,
)


class TestParseTimer:
    """Tests for timer string parsing."""

    def test_standard_format(self) -> None:
        assert parse_timer("4:52") == 292.0

    def test_zero_minutes(self) -> None:
        assert parse_timer("0:30") == 30.0

    def test_exact_minutes(self) -> None:
        assert parse_timer("3:00") == 180.0

    def test_single_digit_seconds(self) -> None:
        assert parse_timer("2:5") == 125.0

    def test_whitespace(self) -> None:
        assert parse_timer("  1:30  ") == 90.0

    def test_invalid_no_colon(self) -> None:
        assert parse_timer("430") is None

    def test_invalid_empty(self) -> None:
        assert parse_timer("") is None

    def test_invalid_letters(self) -> None:
        assert parse_timer("a:bc") is None

    def test_invalid_seconds_60(self) -> None:
        assert parse_timer("1:60") is None

    def test_five_minutes(self) -> None:
        assert parse_timer("5:00") == 300.0


class TestDetermineRule:
    """Tests for rule type determination."""

    def test_5min_rule_high_timer(self) -> None:
        duration, dtype = determine_rule(292.0)  # 4:52
        assert duration == 300
        assert dtype == "5min"

    def test_5min_rule_just_above_3min(self) -> None:
        duration, dtype = determine_rule(181.0)
        assert duration == 300
        assert dtype == "5min"

    def test_3min_rule_at_boundary(self) -> None:
        duration, dtype = determine_rule(180.0)  # exactly 3:00
        assert duration == 180
        assert dtype == "3min"

    def test_3min_rule_low_timer(self) -> None:
        duration, dtype = determine_rule(30.0)
        assert duration == 180
        assert dtype == "3min"

    def test_3min_rule_just_below(self) -> None:
        duration, dtype = determine_rule(170.0)  # 2:50
        assert duration == 180
        assert dtype == "3min"


class TestCalcMatchStart:
    """Tests for match start calculation."""

    def test_5min_rule_example(self) -> None:
        """frame_timestamp=120, timer=3:10 -> start = 120 - (300-190) = 10."""
        start = calc_match_start(120.0, 190.0, 300)
        assert start == 10.0

    def test_3min_rule_example(self) -> None:
        """frame_timestamp=120, timer=2:50 -> start = 120 - (180-170) = 110."""
        start = calc_match_start(120.0, 170.0, 180)
        assert start == 110.0

    def test_beginning_of_match(self) -> None:
        """frame_timestamp=5, timer=4:55 -> start = 5 - (300-295) = 0."""
        start = calc_match_start(5.0, 295.0, 300)
        assert start == 0.0

    def test_end_of_match(self) -> None:
        """frame_timestamp=290, timer=0:10 -> start = 290 - (300-10) = 0."""
        start = calc_match_start(290.0, 10.0, 300)
        assert start == 0.0

    def test_negative_start_clamped_by_cluster(self) -> None:
        """Negative start is allowed here; clamped to 0 in cluster_readings."""
        start = calc_match_start(2.0, 295.0, 300)
        assert start == -3.0


class TestClusterReadings:
    """Tests for grouping readings into matches."""

    def _make_reading(
        self,
        frame_ts: float,
        timer_sec: float,
        total: int = 300,
        dtype: str = "5min",
    ) -> TimerReading:
        return TimerReading(
            frame_timestamp=frame_ts,
            timer_seconds=timer_sec,
            total_duration=total,
            duration_type=dtype,
            match_start=calc_match_start(frame_ts, timer_sec, total),
        )

    def test_empty(self) -> None:
        matches, resolved = cluster_readings([])
        assert matches == []
        assert resolved == []

    def test_single_reading(self) -> None:
        readings = [self._make_reading(120.0, 190.0)]
        matches, _ = cluster_readings(readings)
        assert len(matches) == 1
        assert matches[0].duration_type == "5min"
        assert matches[0].duration_seconds == 300

    def test_single_match_multiple_readings(self) -> None:
        """Multiple readings from the same match cluster together."""
        readings = [
            self._make_reading(30.0, 270.0),  # start = 0
            self._make_reading(60.0, 240.0),  # start = 0
            self._make_reading(90.0, 210.0),  # start = 0
            self._make_reading(120.0, 180.1),  # start ~= 0 (within tolerance)
        ]
        matches, _ = cluster_readings(readings)
        assert len(matches) == 1
        assert matches[0].start_seconds == 0.0

    def test_two_matches(self) -> None:
        """Two separate matches detected."""
        readings = [
            # Match 1: starts at ~10s
            self._make_reading(40.0, 270.0),  # start = 40 - 30 = 10
            self._make_reading(70.0, 240.0),  # start = 70 - 60 = 10
            # Match 2: starts at ~400s
            self._make_reading(430.0, 270.0),  # start = 430 - 30 = 400
            self._make_reading(460.0, 240.0),  # start = 460 - 60 = 400
        ]
        matches, _ = cluster_readings(readings)
        assert len(matches) == 2
        assert matches[0].start_seconds == 10.0
        assert matches[1].start_seconds == 400.0

    def test_negative_start_clamped(self) -> None:
        """Negative calculated start is clamped to 0."""
        readings = [self._make_reading(2.0, 295.0)]  # start = -3.0
        matches, _ = cluster_readings(readings)
        assert len(matches) == 1
        assert matches[0].start_seconds == 0.0

    def test_mixed_rules(self) -> None:
        """3min match detected alongside 5min matches."""
        readings = [
            # 5min match at ~0s
            self._make_reading(30.0, 270.0, 300, "5min"),
            self._make_reading(60.0, 240.0, 300, "5min"),
            self._make_reading(90.0, 210.0, 300, "5min"),
            # 3min match at ~600s (far enough to not cluster with 5min)
            # timer=170: 5min start=600-130=470, 3min start=600-10=590
            # timer=140: 5min start=630-160=470, 3min start=630-40=590
            # 3min cluster(590) has 2, 5min cluster(470) has 2
            # but 5min match0(start=0) has 3 members — no confusion
            self._make_reading(600.0, 170.0, 180, "3min"),
            self._make_reading(630.0, 140.0, 180, "3min"),
            self._make_reading(660.0, 110.0, 180, "3min"),
        ]
        matches, _ = cluster_readings(readings)
        assert len(matches) == 2
        assert matches[0].duration_type == "5min"
        assert matches[1].duration_type == "3min"
        assert matches[1].duration_seconds == 180

    def test_sorted_output(self) -> None:
        """Output is sorted by start_seconds even when input is not."""
        readings = [
            self._make_reading(430.0, 270.0),  # start = 400
            self._make_reading(40.0, 270.0),  # start = 10
        ]
        matches, _ = cluster_readings(readings)
        assert matches[0].start_seconds < matches[1].start_seconds

    def test_ambiguous_timer_resolves_to_5min(self) -> None:
        """Timer <= 3:00 in a 5min match should cluster with 5min readings."""
        readings = [
            # 5min match at ~14s: timer > 3:00
            self._make_reading(20.0, 294.0),  # start = 20-(300-294) = 14
            self._make_reading(40.0, 274.0),  # start = 14
            self._make_reading(60.0, 254.0),  # start = 14
            # Same match but timer <= 3:00 (ambiguous)
            self._make_reading(160.0, 154.0),  # 5min: start=14, 3min: start=134
            self._make_reading(180.0, 134.0),  # 5min: start=14, 3min: start=134
        ]
        matches, resolved = cluster_readings(readings)
        assert len(matches) == 1
        assert matches[0].duration_type == "5min"
        assert matches[0].start_seconds == 14.0
        assert len(resolved) == 5
