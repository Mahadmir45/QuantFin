"""
Configuration management for MMM Bayesian models.

Handles YAML-based configuration loading, validation, and defaults
for model hyperparameters, channel definitions, and optimization constraints.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChannelConfig:
    """Configuration for a single media/marketing channel."""

    name: str
    adstock_type: str = "geometric"       # geometric | weibull_cdf | weibull_pdf | delayed
    adstock_max_lag: int = 8
    adstock_alpha_prior: tuple[float, float] = (2.0, 2.0)  # Beta prior
    saturation_type: str = "hill"          # hill | logistic | s_curve
    saturation_lambda_prior: tuple[float, float] = (0.5, 1.0)  # HalfNormal σ
    saturation_k_prior: tuple[float, float] = (1.0, 1.0)       # Gamma α,β
    spend_share_prior: float | None = None
    min_budget_pct: float = 0.0
    max_budget_pct: float = 1.0


@dataclass
class SeasonalityConfig:
    """Fourier seasonality configuration."""

    yearly_order: int = 6
    quarterly_order: int = 2
    monthly_order: int = 0
    custom_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Full model configuration."""

    # Target
    target_col: str = "revenue"
    date_col: str = "date"
    target_transform: str = "log"  # log | none | sqrt

    # Channels
    channels: list[ChannelConfig] = field(default_factory=list)

    # Control variables
    control_cols: list[str] = field(default_factory=list)

    # Seasonality
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)

    # Trend
    trend_type: str = "linear"  # linear | logistic | rw (random walk)

    # Priors
    intercept_prior: tuple[float, float] = (0.0, 2.0)  # Normal μ, σ
    noise_prior: float = 1.0                             # HalfNormal σ
    channel_coeff_prior: float = 0.5                     # HalfNormal σ

    # Sampler
    n_chains: int = 4
    n_draws: int = 2000
    n_tune: int = 1000
    target_accept: float = 0.9
    random_seed: int = 42

    # Validation
    holdout_pct: float = 0.15

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        channels = [
            ChannelConfig(**ch) for ch in raw.pop("channels", [])
        ]
        seasonality = SeasonalityConfig(**raw.pop("seasonality", {}))

        return cls(channels=channels, seasonality=seasonality, **raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize configuration to YAML."""
        data = self.__dict__.copy()
        data["channels"] = [ch.__dict__ for ch in self.channels]
        data["seasonality"] = self.seasonality.__dict__
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> list[str]:
        """Return a list of validation warnings/errors."""
        issues: list[str] = []
        if not self.channels:
            issues.append("No channels defined — model will have no media variables.")
        if self.holdout_pct <= 0 or self.holdout_pct >= 1:
            issues.append("holdout_pct must be in (0, 1).")
        if self.n_draws < 500:
            issues.append("n_draws < 500 may yield unreliable posteriors.")
        for ch in self.channels:
            if ch.adstock_type not in ("geometric", "weibull_cdf", "weibull_pdf", "delayed"):
                issues.append(f"Channel '{ch.name}': unknown adstock_type '{ch.adstock_type}'")
            if ch.saturation_type not in ("hill", "logistic", "s_curve"):
                issues.append(f"Channel '{ch.name}': unknown saturation_type '{ch.saturation_type}'")
        return issues
