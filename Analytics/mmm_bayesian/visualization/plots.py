"""
Publication-quality visualization suite for MMM results.

Generates the key charts that Ekimetrics-style analytics deliverables
include: waterfall decomposition, response curves, ROI heatmaps,
budget allocation comparisons, posterior diagnostics, and time-series
fit plots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# House style
PALETTE = [
    "#1B2A4A", "#2E86AB", "#A23B72", "#F18F01",
    "#C73E1D", "#3B1F2B", "#44BBA4", "#E94F37",
    "#393E41", "#8D6A9F", "#D4A373", "#3A86FF",
]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ===================================================================
# 1. Revenue decomposition waterfall
# ===================================================================

def plot_decomposition_waterfall(
    decomposition_df: pd.DataFrame,
    channel_cols: list[str] | None = None,
    output_dir: str | Path = "outputs/plots",
    title: str = "Revenue Decomposition — Contribution Waterfall",
) -> plt.Figure:
    """
    Waterfall chart showing average contribution of each component.
    """
    output_dir = _ensure_dir(output_dir)

    if channel_cols is None:
        channel_cols = [c for c in decomposition_df.columns if c.startswith("media_")]

    components = ["intercept", "trend", "seasonality"] + channel_cols
    values = [decomposition_df[c].mean() for c in components if c in decomposition_df.columns]
    labels = [c.replace("media_", "").replace("_", " ").title() for c in components if c in decomposition_df.columns]

    cumulative = np.cumsum(values)
    starts = np.concatenate([[0], cumulative[:-1]])

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]

    bars = ax.bar(labels, values, bottom=starts, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        y_pos = bar.get_y() + bar.get_height() / 2
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:+.2f}", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Contribution (scaled)")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "decomposition_waterfall.png")
    logger.info("Saved decomposition waterfall to %s", output_dir)
    return fig


# ===================================================================
# 2. Time-series decomposition
# ===================================================================

def plot_timeseries_decomposition(
    decomposition_df: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> plt.Figure:
    """Stacked area chart of revenue components over time."""
    output_dir = _ensure_dir(output_dir)

    media_cols = [c for c in decomposition_df.columns if c.startswith("media_")]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Panel 1: Actual vs Predicted
    ax = axes[0]
    ax.plot(decomposition_df["date"], decomposition_df["actual"], label="Actual", color=PALETTE[0], linewidth=1.5)
    ax.plot(decomposition_df["date"], decomposition_df["predicted"], label="Predicted", color=PALETTE[1], linewidth=1.5, linestyle="--")
    ax.fill_between(decomposition_df["date"], decomposition_df["actual"], decomposition_df["predicted"], alpha=0.15, color=PALETTE[1])
    ax.set_title("Actual vs Predicted Revenue", fontweight="bold")
    ax.legend(loc="upper left")

    # Panel 2: Base components
    ax = axes[1]
    ax.fill_between(decomposition_df["date"], 0, decomposition_df["intercept"], label="Base", alpha=0.6, color=PALETTE[0])
    ax.fill_between(decomposition_df["date"], decomposition_df["intercept"],
                    decomposition_df["intercept"] + decomposition_df["trend"],
                    label="Trend", alpha=0.6, color=PALETTE[2])
    ax.set_title("Base + Trend", fontweight="bold")
    ax.legend(loc="upper left")

    # Panel 3: Channel contributions stacked
    ax = axes[2]
    bottom = np.zeros(len(decomposition_df))
    for i, col in enumerate(media_cols):
        vals = decomposition_df[col].values
        ax.fill_between(
            decomposition_df["date"], bottom, bottom + vals,
            label=col.replace("media_", "").replace("_", " ").title(),
            alpha=0.7, color=PALETTE[i % len(PALETTE)],
        )
        bottom += vals
    ax.set_title("Channel Contributions Over Time", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "timeseries_decomposition.png")
    logger.info("Saved timeseries decomposition to %s", output_dir)
    return fig


# ===================================================================
# 3. Channel ROI comparison
# ===================================================================

def plot_roi_comparison(
    roi_df: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> plt.Figure:
    """Horizontal bar chart of channel ROI with credible intervals."""
    output_dir = _ensure_dir(output_dir)

    roi_df = roi_df.sort_values("roi_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(roi_df) * 0.8)))

    y_pos = np.arange(len(roi_df))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(roi_df))]

    ax.barh(y_pos, roi_df["roi_mean"], color=colors, edgecolor="white", height=0.6)

    if "roi_5pct" in roi_df.columns and "roi_95pct" in roi_df.columns:
        xerr = np.array([
            roi_df["roi_mean"] - roi_df["roi_5pct"],
            roi_df["roi_95pct"] - roi_df["roi_mean"],
        ])
        xerr = np.maximum(xerr, 0)
        ax.errorbar(
            roi_df["roi_mean"], y_pos, xerr=xerr,
            fmt="none", ecolor="black", capsize=4, linewidth=1.5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(roi_df["channel"].str.replace("_", " ").str.title())
    ax.set_xlabel("Return on Investment (ROI)")
    ax.set_title("Channel ROI Comparison (90% Credible Interval)", fontweight="bold", fontsize=14)

    for i, (_, row) in enumerate(roi_df.iterrows()):
        ax.text(row["roi_mean"] + 0.01, i, f'{row["roi_mean"]:.3f}', va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "roi_comparison.png")
    logger.info("Saved ROI comparison to %s", output_dir)
    return fig


# ===================================================================
# 4. Response curves
# ===================================================================

def plot_response_curves(
    response_df: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> plt.Figure:
    """Per-channel response curves showing diminishing returns."""
    output_dir = _ensure_dir(output_dir)

    channels = response_df["channel"].unique()
    n_ch = len(channels)
    ncols = min(3, n_ch)
    nrows = (n_ch + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, ch in enumerate(channels):
        ax = axes[idx // ncols][idx % ncols]
        ch_data = response_df[response_df["channel"] == ch]
        ax.plot(
            ch_data["weekly_spend"], ch_data["total_response"],
            color=PALETTE[idx % len(PALETTE)], linewidth=2,
        )
        ax.fill_between(ch_data["weekly_spend"], 0, ch_data["total_response"], alpha=0.15, color=PALETTE[idx % len(PALETTE)])
        ax.set_title(ch.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Weekly Spend")
        ax.set_ylabel("Response")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Hide empty subplots
    for idx in range(n_ch, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Channel Response Curves (Diminishing Returns)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "response_curves.png")
    logger.info("Saved response curves to %s", output_dir)
    return fig


# ===================================================================
# 5. Budget allocation comparison
# ===================================================================

def plot_budget_allocation(
    current_allocation: dict[str, float],
    optimal_df: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> plt.Figure:
    """Side-by-side donut charts: current vs optimal allocation."""
    output_dir = _ensure_dir(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    channels = optimal_df["channel"].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(channels))]
    labels = [c.replace("_", " ").title() for c in channels]

    # Current
    current_vals = [current_allocation.get(ch, 0) for ch in channels]
    current_total = sum(current_vals) or 1
    current_pcts = [v / current_total * 100 for v in current_vals]

    wedges1, _, autotexts1 = ax1.pie(
        current_pcts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax1.add_artist(centre_circle)
    ax1.set_title("Current Allocation", fontweight="bold", fontsize=13)

    # Optimal
    optimal_pcts = optimal_df["optimal_pct"].tolist()
    wedges2, _, autotexts2 = ax2.pie(
        optimal_pcts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    centre_circle2 = plt.Circle((0, 0), 0.50, fc="white")
    ax2.add_artist(centre_circle2)
    ax2.set_title("Optimized Allocation", fontweight="bold", fontsize=13)

    fig.suptitle("Budget Allocation: Current vs Optimal", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "budget_allocation.png")
    logger.info("Saved budget allocation to %s", output_dir)
    return fig


# ===================================================================
# 6. Posterior diagnostics
# ===================================================================

def plot_posterior_diagnostics(
    trace: az.InferenceData,
    var_names: list[str] | None = None,
    output_dir: str | Path = "outputs/plots",
) -> plt.Figure:
    """Trace plots and posterior distributions for key parameters."""
    output_dir = _ensure_dir(output_dir)

    if var_names is None:
        var_names = [v for v in trace.posterior.data_vars if not any(
            v.startswith(p) for p in ["contribution", "mu", "trend_comp", "seasonality_comp", "media_comp"]
        )]
        var_names = var_names[:12]

    fig = plt.figure(figsize=(16, 3 * len(var_names)))
    axes = az.plot_trace(trace, var_names=var_names, compact=True, figsize=(16, 3 * len(var_names)))
    plt.suptitle("Posterior Diagnostics", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "posterior_diagnostics.png")
    logger.info("Saved posterior diagnostics to %s", output_dir)
    return fig


# ===================================================================
# 7. Scenario analysis
# ===================================================================

def plot_scenario_analysis(
    scenario_df: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> plt.Figure:
    """Line chart showing expected contribution at different budget levels."""
    output_dir = _ensure_dir(output_dir)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        scenario_df["budget_multiplier"] * 100,
        scenario_df["total_expected_contribution"],
        marker="o", linewidth=2.5, color=PALETTE[1], markersize=8,
    )
    ax1.set_xlabel("Budget Level (% of Current)", fontsize=12)
    ax1.set_ylabel("Expected Total Contribution", fontsize=12, color=PALETTE[1])
    ax1.tick_params(axis="y", labelcolor=PALETTE[1])

    ax2 = ax1.twinx()
    ax2.plot(
        scenario_df["budget_multiplier"] * 100,
        scenario_df["marginal_roi"],
        marker="s", linewidth=2.5, color=PALETTE[3], linestyle="--", markersize=8,
    )
    ax2.set_ylabel("Marginal ROI", fontsize=12, color=PALETTE[3])
    ax2.tick_params(axis="y", labelcolor=PALETTE[3])

    ax1.set_title("Scenario Analysis: Budget Level Impact", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "scenario_analysis.png")
    logger.info("Saved scenario analysis to %s", output_dir)
    return fig


# ===================================================================
# 8. Spend vs contribution share
# ===================================================================

def plot_spend_vs_contribution(
    roi_df: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> plt.Figure:
    """Grouped bar chart: spend share vs contribution share per channel."""
    output_dir = _ensure_dir(output_dir)

    total_spend = roi_df["total_spend"].sum()
    total_contrib = roi_df["total_contribution_mean"].sum()

    roi_df = roi_df.copy()
    roi_df["spend_share"] = roi_df["total_spend"] / total_spend * 100
    roi_df["contribution_share"] = roi_df["total_contribution_mean"] / total_contrib * 100

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(roi_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, roi_df["spend_share"], width, label="Spend Share %", color=PALETTE[0])
    bars2 = ax.bar(x + width / 2, roi_df["contribution_share"], width, label="Contribution Share %", color=PALETTE[1])

    ax.set_xticks(x)
    ax.set_xticklabels(roi_df["channel"].str.replace("_", " ").str.title(), rotation=35, ha="right")
    ax.set_ylabel("Share (%)")
    ax.set_title("Spend Share vs Revenue Contribution Share", fontweight="bold", fontsize=14)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "spend_vs_contribution.png")
    logger.info("Saved spend vs contribution to %s", output_dir)
    return fig
