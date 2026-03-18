"""
Synthetic data generator for MMM demonstration and testing.

Generates realistic multi-channel marketing data with known ground-truth
parameters so that model recovery can be validated. Supports retail,
luxury, and DTC brand archetypes with industry-typical channel mixes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ChannelSpec:
    """Ground-truth specification for one media channel."""

    name: str
    weekly_mean_spend: float
    weekly_std_spend: float
    true_beta: float          # coefficient on saturated-adstocked variable
    adstock_alpha: float      # geometric decay rate [0, 1)
    saturation_lam: float     # Hill half-saturation
    saturation_k: float       # Hill steepness
    max_lag: int = 8


@dataclass
class ScenarioSpec:
    """Full scenario specification."""

    name: str
    n_weeks: int = 156  # 3 years
    base_revenue: float = 500_000.0
    trend_slope: float = 200.0
    noise_std: float = 15_000.0
    channels: list[ChannelSpec] = field(default_factory=list)
    seed: int = 42


# ---------------------------------------------------------------------------
# Pre-built industry archetypes
# ---------------------------------------------------------------------------

RETAIL_SCENARIO = ScenarioSpec(
    name="retail_omnichannel",
    n_weeks=156,
    base_revenue=800_000,
    trend_slope=350,
    noise_std=20_000,
    channels=[
        ChannelSpec("tv",            weekly_mean_spend=120_000, weekly_std_spend=30_000,
                    true_beta=0.30, adstock_alpha=0.70, saturation_lam=0.5, saturation_k=2.0),
        ChannelSpec("digital_video", weekly_mean_spend=60_000,  weekly_std_spend=15_000,
                    true_beta=0.22, adstock_alpha=0.50, saturation_lam=0.4, saturation_k=2.5),
        ChannelSpec("paid_search",   weekly_mean_spend=45_000,  weekly_std_spend=12_000,
                    true_beta=0.25, adstock_alpha=0.15, saturation_lam=0.3, saturation_k=3.0),
        ChannelSpec("paid_social",   weekly_mean_spend=35_000,  weekly_std_spend=10_000,
                    true_beta=0.18, adstock_alpha=0.40, saturation_lam=0.35, saturation_k=2.2),
        ChannelSpec("display",       weekly_mean_spend=25_000,  weekly_std_spend=8_000,
                    true_beta=0.10, adstock_alpha=0.55, saturation_lam=0.45, saturation_k=1.8),
        ChannelSpec("print",         weekly_mean_spend=15_000,  weekly_std_spend=5_000,
                    true_beta=0.06, adstock_alpha=0.60, saturation_lam=0.5, saturation_k=1.5),
    ],
)

LUXURY_SCENARIO = ScenarioSpec(
    name="luxury_brand",
    n_weeks=156,
    base_revenue=2_000_000,
    trend_slope=500,
    noise_std=50_000,
    channels=[
        ChannelSpec("tv_prestige",   weekly_mean_spend=200_000, weekly_std_spend=50_000,
                    true_beta=0.25, adstock_alpha=0.75, saturation_lam=0.6, saturation_k=1.5),
        ChannelSpec("digital_video", weekly_mean_spend=80_000,  weekly_std_spend=20_000,
                    true_beta=0.20, adstock_alpha=0.45, saturation_lam=0.4, saturation_k=2.0),
        ChannelSpec("influencer",    weekly_mean_spend=50_000,  weekly_std_spend=15_000,
                    true_beta=0.28, adstock_alpha=0.35, saturation_lam=0.3, saturation_k=2.8),
        ChannelSpec("paid_social",   weekly_mean_spend=40_000,  weekly_std_spend=10_000,
                    true_beta=0.15, adstock_alpha=0.40, saturation_lam=0.35, saturation_k=2.5),
        ChannelSpec("ooh_premium",   weekly_mean_spend=60_000,  weekly_std_spend=20_000,
                    true_beta=0.12, adstock_alpha=0.65, saturation_lam=0.5, saturation_k=1.8),
        ChannelSpec("print_magazine",weekly_mean_spend=30_000,  weekly_std_spend=10_000,
                    true_beta=0.08, adstock_alpha=0.70, saturation_lam=0.55, saturation_k=1.3),
    ],
)

DTC_SCENARIO = ScenarioSpec(
    name="dtc_ecommerce",
    n_weeks=104,
    base_revenue=300_000,
    trend_slope=800,
    noise_std=12_000,
    channels=[
        ChannelSpec("paid_search",   weekly_mean_spend=25_000, weekly_std_spend=8_000,
                    true_beta=0.35, adstock_alpha=0.10, saturation_lam=0.25, saturation_k=3.5),
        ChannelSpec("paid_social",   weekly_mean_spend=30_000, weekly_std_spend=10_000,
                    true_beta=0.30, adstock_alpha=0.35, saturation_lam=0.3, saturation_k=2.5),
        ChannelSpec("email",         weekly_mean_spend=5_000,  weekly_std_spend=1_500,
                    true_beta=0.20, adstock_alpha=0.05, saturation_lam=0.2, saturation_k=4.0),
        ChannelSpec("affiliate",     weekly_mean_spend=10_000, weekly_std_spend=3_000,
                    true_beta=0.15, adstock_alpha=0.20, saturation_lam=0.3, saturation_k=2.0),
        ChannelSpec("display_retarget", weekly_mean_spend=12_000, weekly_std_spend=4_000,
                    true_beta=0.22, adstock_alpha=0.30, saturation_lam=0.25, saturation_k=3.0),
    ],
)

SCENARIOS: dict[str, ScenarioSpec] = {
    "retail": RETAIL_SCENARIO,
    "luxury": LUXURY_SCENARIO,
    "dtc": DTC_SCENARIO,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _geometric_adstock(x: np.ndarray, alpha: float, max_lag: int) -> np.ndarray:
    """Apply geometric (carry-over) adstock transformation."""
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + alpha * out[t - 1]
    return out


def _hill_saturation(x: np.ndarray, lam: float, k: float) -> np.ndarray:
    """Apply Hill function saturation: x^k / (lam^k + x^k)."""
    xk = np.power(np.maximum(x, 0), k)
    return xk / (np.power(lam, k) + xk + 1e-12)


def generate_synthetic_data(
    scenario: ScenarioSpec | str = "retail",
) -> tuple[pd.DataFrame, dict]:
    """
    Generate a synthetic weekly marketing dataset with known ground truth.

    Parameters
    ----------
    scenario : A ScenarioSpec or one of 'retail', 'luxury', 'dtc'.

    Returns
    -------
    df : DataFrame with date, channel spends, controls, and revenue.
    ground_truth : Dict of true parameters for model validation.
    """
    if isinstance(scenario, str):
        scenario = SCENARIOS[scenario]

    rng = np.random.default_rng(scenario.seed)
    n = scenario.n_weeks

    dates = pd.date_range(start="2021-01-04", periods=n, freq="W-MON")
    df = pd.DataFrame({"date": dates})

    # Trend
    trend = scenario.trend_slope * np.arange(n) / n

    # Seasonality (annual + holiday spikes)
    t_idx = np.arange(n)
    seasonality = (
        0.08 * np.sin(2 * np.pi * t_idx / 52)
        + 0.04 * np.cos(4 * np.pi * t_idx / 52)
        + 0.06 * np.sin(6 * np.pi * t_idx / 52)
    )
    # Holiday weeks (Black Friday ~week 47, Christmas ~week 51-52)
    week_of_year = dates.isocalendar().week.values.astype(int)
    holiday_lift = np.where(np.isin(week_of_year, [47, 48, 51, 52]), 0.15, 0.0)
    # Summer dip for luxury
    if "luxury" in scenario.name:
        holiday_lift += np.where(np.isin(week_of_year, list(range(28, 35))), -0.05, 0.0)

    # Control: competitor price index
    competitor_idx = 1.0 + 0.05 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    df["competitor_price_index"] = competitor_idx
    control_effect = -0.03 * (competitor_idx - 1.0)

    # Temperature (proxy for foot traffic in retail)
    temperature = 15 + 12 * np.sin(2 * np.pi * (t_idx - 10) / 52) + 2 * rng.standard_normal(n)
    df["temperature"] = temperature
    temp_effect = 0.01 * (temperature - temperature.mean()) / temperature.std()

    # Generate channel spends and compute media contribution
    media_contribution = np.zeros(n)
    ground_truth: dict = {"channels": {}, "base_revenue": scenario.base_revenue}

    for ch in scenario.channels:
        raw_spend = np.maximum(
            rng.normal(ch.weekly_mean_spend, ch.weekly_std_spend, n), 0
        )
        # Brands often pulse spend — add some zero-spend weeks
        if rng.random() < 0.3:
            zero_mask = rng.random(n) < 0.15
            raw_spend[zero_mask] = 0.0

        df[f"spend_{ch.name}"] = raw_spend

        # Normalize for adstock + saturation
        spend_norm = raw_spend / (raw_spend.max() + 1e-12)
        adstocked = _geometric_adstock(spend_norm, ch.adstock_alpha, ch.max_lag)
        saturated = _hill_saturation(adstocked, ch.saturation_lam, ch.saturation_k)

        media_contribution += ch.true_beta * saturated * scenario.base_revenue

        ground_truth["channels"][ch.name] = {
            "true_beta": ch.true_beta,
            "adstock_alpha": ch.adstock_alpha,
            "saturation_lam": ch.saturation_lam,
            "saturation_k": ch.saturation_k,
            "total_spend": float(raw_spend.sum()),
            "total_contribution": float((ch.true_beta * saturated * scenario.base_revenue).sum()),
        }

    # Assemble revenue
    noise = rng.normal(0, scenario.noise_std, n)
    revenue = (
        scenario.base_revenue
        + trend * scenario.base_revenue
        + seasonality * scenario.base_revenue
        + holiday_lift * scenario.base_revenue
        + control_effect * scenario.base_revenue
        + temp_effect * scenario.base_revenue
        + media_contribution
        + noise
    )
    df["revenue"] = np.maximum(revenue, 0)

    ground_truth["trend_slope"] = scenario.trend_slope
    ground_truth["noise_std"] = scenario.noise_std
    ground_truth["total_media_contribution_pct"] = float(
        media_contribution.sum() / revenue.sum() * 100
    )

    return df, ground_truth
