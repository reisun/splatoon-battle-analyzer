"""Scoring configuration loader."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring.yaml"


@dataclass(frozen=True)
class ScoringWeights:
    kills: float = 1.0
    score_gain: float = 1.0
    special: float = 1.0


@dataclass(frozen=True)
class ScoringConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    death_penalty: float = 0.5
    score_gain_window_seconds: int = 30


def load_scoring_config(path: Path | None = None) -> ScoringConfig:
    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return ScoringConfig()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return ScoringConfig()
    w = raw.get("weights", {})
    weights = ScoringWeights(
        kills=float(w.get("kills", 1.0)),
        score_gain=float(w.get("score_gain", 1.0)),
        special=float(w.get("special", 1.0)),
    )
    return ScoringConfig(
        weights=weights,
        death_penalty=float(raw.get("death_penalty", 0.5)),
        score_gain_window_seconds=int(raw.get("score_gain_window_seconds", 30)),
    )
