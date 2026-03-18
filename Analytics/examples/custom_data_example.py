"""
Custom Data Example
===================

Shows how to bring your own data and configure the model
for a custom channel mix. This template is the starting point
for real-world engagements.

Expected CSV format:
    date, spend_channel1, spend_channel2, ..., control1, ..., revenue

Run:
    cd Analytics/
    python examples/custom_data_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmm_bayesian.core.config import ModelConfig, ChannelConfig, SeasonalityConfig
from mmm_bayesian.models.bayesian_mmm import BayesianMMM
from mmm_bayesian.optimization.budget_optimizer import BudgetOptimizer, extract_posterior_params
from mmm_bayesian.data.synthetic import generate_synthetic_data


def main():
    # --- Generate demo data (replace with pd.read_csv("your_data.csv")) ---
    df, _ = generate_synthetic_data("dtc")

    # --- Define your custom config ---
    config = ModelConfig(
        target_col="revenue",
        date_col="date",
        target_transform="log",
        channels=[
            ChannelConfig(name="paid_search", adstock_type="geometric", adstock_max_lag=3,
                          saturation_type="hill"),
            ChannelConfig(name="paid_social", adstock_type="geometric", adstock_max_lag=5,
                          saturation_type="hill"),
            ChannelConfig(name="email", adstock_type="geometric", adstock_max_lag=2,
                          saturation_type="hill"),
            ChannelConfig(name="affiliate", adstock_type="geometric", adstock_max_lag=4,
                          saturation_type="hill"),
            ChannelConfig(name="display_retarget", adstock_type="geometric", adstock_max_lag=4,
                          saturation_type="hill"),
        ],
        control_cols=["competitor_price_index", "temperature"],
        seasonality=SeasonalityConfig(yearly_order=4),
        trend_type="linear",
        n_chains=2,
        n_draws=500,
        n_tune=300,
        holdout_pct=0.15,
    )

    # --- Fit ---
    mmm = BayesianMMM(config)
    mmm.fit(df)

    # --- Analyze ---
    print("\nParameter Summary:")
    print(mmm.summary().to_string())

    print("\nChannel ROI:")
    roi = mmm.compute_roi()
    print(roi.to_string())

    print("\nValidation:")
    print(mmm.validate())

    # --- Optimize ---
    responses = extract_posterior_params(mmm)
    total_budget = sum(df[f"spend_{ch.name}"].sum() for ch in config.channels)
    optimizer = BudgetOptimizer(responses, total_budget)

    print("\nOptimal Allocation:")
    print(optimizer.optimize().to_string())


if __name__ == "__main__":
    main()
