"""Tests for the scoring configuration loader."""

from pathlib import Path
from textwrap import dedent

from src.scoring_config import ScoringConfig, ScoringWeights, load_scoring_config, override_weights


class TestLoadScoringConfig:
    """Tests for load_scoring_config."""

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_scoring_config(tmp_path / "nonexistent.yaml")
        assert cfg == ScoringConfig()

    def test_loads_custom_values(self, tmp_path: Path) -> None:
        p = tmp_path / "scoring.yaml"
        p.write_text(
            dedent("""\
                weights:
                  kills: 2.0
                  score_count_gain: 0.8
                death_penalty: 0.3
                score_count_gain_window_seconds: 20
            """)
        )
        cfg = load_scoring_config(p)
        assert cfg.weights.kills == 2.0
        assert cfg.weights.score_count_gain == 0.8
        assert cfg.death_penalty == 0.3
        assert cfg.score_count_gain_window_seconds == 20

    def test_partial_weights_default_rest(self, tmp_path: Path) -> None:
        p = tmp_path / "scoring.yaml"
        p.write_text("weights:\n  kills: 1.5\n")
        cfg = load_scoring_config(p)
        assert cfg.weights.kills == 1.5
        assert cfg.death_penalty == 0.5

    def test_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "scoring.yaml"
        p.write_text("")
        cfg = load_scoring_config(p)
        assert cfg == ScoringConfig()


class TestScoringDefaults:
    """Tests for default config values."""

    def test_default_weights(self) -> None:
        w = ScoringWeights()
        assert w.kills == 1.0
        assert w.score_count_gain == 1.0

    def test_default_config(self) -> None:
        cfg = ScoringConfig()
        assert cfg.death_penalty == 0.5
        assert cfg.score_count_gain_window_seconds == 30


class TestOverrideWeights:
    """Tests for override_weights."""

    def test_override_score_count_gain_to_zero(self) -> None:
        cfg = ScoringConfig()
        overridden = override_weights(cfg, {"score_count_gain": 0})
        assert overridden.weights.score_count_gain == 0
        assert overridden.weights.kills == 1.0
        assert overridden.death_penalty == 0.5

    def test_override_kills(self) -> None:
        cfg = ScoringConfig()
        overridden = override_weights(cfg, {"kills": 5.0})
        assert overridden.weights.kills == 5.0
        assert overridden.weights.score_count_gain == 1.0

    def test_empty_overrides_returns_same(self) -> None:
        cfg = ScoringConfig()
        result = override_weights(cfg, {})
        assert result is cfg

    def test_none_overrides_returns_same(self) -> None:
        cfg = ScoringConfig()
        result = override_weights(cfg, {})
        assert result is cfg
