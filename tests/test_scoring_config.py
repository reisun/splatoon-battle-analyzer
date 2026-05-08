"""Tests for the scoring configuration loader."""

from pathlib import Path
from textwrap import dedent

from src.scoring_config import ScoringConfig, ScoringWeights, load_scoring_config


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
