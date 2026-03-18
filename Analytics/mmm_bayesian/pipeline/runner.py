"""
End-to-end MMM pipeline runner.

Orchestrates the full workflow:
    1. Data loading (CSV or synthetic generation)
    2. Configuration loading / defaults
    3. Model building and fitting
    4. Decomposition and ROI computation
    5. Budget optimization
    6. Visualization and report generation
    7. Artifact export (trace, tables, plots)

Can be invoked programmatically or via CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from ..core.config import ModelConfig, ChannelConfig
from ..models.bayesian_mmm import BayesianMMM
from ..data.synthetic import generate_synthetic_data
from ..optimization.budget_optimizer import BudgetOptimizer, extract_posterior_params
from ..visualization.plots import (
    plot_decomposition_waterfall,
    plot_timeseries_decomposition,
    plot_roi_comparison,
    plot_response_curves,
    plot_budget_allocation,
    plot_scenario_analysis,
    plot_spend_vs_contribution,
    plot_posterior_diagnostics,
)

logger = logging.getLogger(__name__)


def _default_config_for_scenario(scenario_name: str) -> ModelConfig:
    """Build a sensible default ModelConfig for a named scenario."""
    from ..data.synthetic import SCENARIOS
    scenario = SCENARIOS[scenario_name]

    channels = [
        ChannelConfig(
            name=ch.name,
            adstock_type="geometric",
            adstock_max_lag=ch.max_lag,
            saturation_type="hill",
        )
        for ch in scenario.channels
    ]

    return ModelConfig(
        target_col="revenue",
        date_col="date",
        target_transform="log",
        channels=channels,
        control_cols=["competitor_price_index", "temperature"],
        trend_type="linear",
        n_chains=4,
        n_draws=2000,
        n_tune=1000,
        target_accept=0.9,
        holdout_pct=0.15,
    )


def run_pipeline(
    data_path: str | None = None,
    config_path: str | None = None,
    scenario: str = "retail",
    output_dir: str = "outputs",
    quick_mode: bool = False,
) -> dict:
    """
    Execute the full MMM pipeline.

    Parameters
    ----------
    data_path : Path to CSV data. If None, generates synthetic data.
    config_path : Path to YAML config. If None, uses scenario defaults.
    scenario : One of 'retail', 'luxury', 'dtc'. Used when data_path is None.
    output_dir : Directory for all outputs (plots, tables, artifacts).
    quick_mode : If True, use fewer draws for faster iteration.

    Returns
    -------
    Dict with keys: metrics, roi, decomposition, optimization, scenario_analysis.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir) / f"run_{run_id}"
    plot_dir = out_root / "plots"
    table_dir = out_root / "tables"
    for d in [out_root, plot_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_root / "pipeline.log"),
        ],
    )

    # ---- Step 1: Load data ----
    if data_path is not None:
        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path)
        ground_truth = None
    else:
        logger.info("Generating synthetic '%s' scenario data", scenario)
        df, ground_truth = generate_synthetic_data(scenario)
        df.to_csv(table_dir / "synthetic_data.csv", index=False)
        if ground_truth:
            with open(table_dir / "ground_truth.json", "w") as f:
                json.dump(ground_truth, f, indent=2, default=str)

    logger.info("Data shape: %s", df.shape)

    # ---- Step 2: Load config ----
    if config_path is not None:
        config = ModelConfig.from_yaml(config_path)
    else:
        config = _default_config_for_scenario(scenario)

    if quick_mode:
        config.n_draws = 500
        config.n_tune = 300
        config.n_chains = 2

    issues = config.validate()
    if issues:
        logger.warning("Config validation issues: %s", issues)

    config.to_yaml(out_root / "config.yaml")

    # ---- Step 3: Build & fit model ----
    logger.info("=" * 60)
    logger.info("BUILDING BAYESIAN MMM")
    logger.info("=" * 60)

    mmm = BayesianMMM(config)
    trace = mmm.fit(df)

    # ---- Step 4: Diagnostics ----
    summary_df = mmm.summary()
    summary_df.to_csv(table_dir / "parameter_summary.csv")
    logger.info("\n%s", summary_df.to_string())

    try:
        plot_posterior_diagnostics(trace, output_dir=plot_dir)
    except Exception as e:
        logger.warning("Could not generate posterior diagnostics plot: %s", e)

    # ---- Step 5: Decomposition ----
    decomposition_df = mmm.decompose()
    decomposition_df.to_csv(table_dir / "decomposition.csv", index=False)

    plot_decomposition_waterfall(decomposition_df, output_dir=plot_dir)
    plot_timeseries_decomposition(decomposition_df, output_dir=plot_dir)

    # ---- Step 6: ROI ----
    roi_df = mmm.compute_roi()
    roi_df.to_csv(table_dir / "channel_roi.csv", index=False)
    logger.info("Channel ROI:\n%s", roi_df.to_string())

    plot_roi_comparison(roi_df, output_dir=plot_dir)
    plot_spend_vs_contribution(roi_df, output_dir=plot_dir)

    # ---- Step 7: Validation ----
    metrics = mmm.validate()
    logger.info("Validation Metrics: %s", metrics)
    with open(table_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Step 8: Budget Optimization ----
    channel_responses = extract_posterior_params(mmm)
    total_budget = sum(
        df[f"spend_{ch.name}"].sum()
        for ch in config.channels
        if f"spend_{ch.name}" in df.columns
    )
    optimizer = BudgetOptimizer(channel_responses, total_budget)
    optimal_df = optimizer.optimize()
    optimal_df.to_csv(table_dir / "optimal_allocation.csv", index=False)
    logger.info("Optimal Allocation:\n%s", optimal_df.to_string())

    # Current allocation for comparison
    current_alloc = {
        ch.name: df[f"spend_{ch.name}"].sum()
        for ch in config.channels
        if f"spend_{ch.name}" in df.columns
    }
    plot_budget_allocation(current_alloc, optimal_df, output_dir=plot_dir)

    # Response curves
    response_df = optimizer.response_curves()
    response_df.to_csv(table_dir / "response_curves.csv", index=False)
    plot_response_curves(response_df, output_dir=plot_dir)

    # Scenario analysis
    scenario_df = optimizer.scenario_analysis()
    scenario_df.to_csv(table_dir / "scenario_analysis.csv", index=False)
    plot_scenario_analysis(scenario_df, output_dir=plot_dir)

    # ---- Final summary ----
    results = {
        "run_id": run_id,
        "output_dir": str(out_root),
        "metrics": metrics,
        "roi": roi_df.to_dict(orient="records"),
        "optimal_allocation": optimal_df.to_dict(orient="records"),
        "scenario_analysis": scenario_df.to_dict(orient="records"),
    }

    with open(out_root / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — outputs saved to %s", out_root)
    logger.info("=" * 60)

    return results
