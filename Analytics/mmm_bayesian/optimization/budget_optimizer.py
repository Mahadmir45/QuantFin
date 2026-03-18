"""
Budget allocation optimizer for Marketing Mix Models.

Given a fitted Bayesian MMM, this module finds the optimal distribution
of a fixed marketing budget across channels to maximize total expected
revenue (or any KPI). Uses scipy's constrained optimizer (SLSQP) to
solve the nonlinear allocation problem.

Also computes:
    - Marginal ROI (mROI) response curves per channel
    - Optimal vs. current allocation comparison
    - Scenario analysis (budget increase / decrease)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..transforms.adstock import geometric_adstock_np, weibull_adstock_np, delayed_adstock_np
from ..transforms.saturation import hill_saturation_np, logistic_saturation_np, gompertz_saturation_np

logger = logging.getLogger(__name__)


@dataclass
class ChannelResponse:
    """Learned response curve parameters for a single channel."""

    name: str
    adstock_type: str
    adstock_alpha: float
    adstock_max_lag: int
    saturation_type: str
    saturation_lam: float
    saturation_k: float
    beta: float
    weibull_shape: float = 1.0
    weibull_scale: float = 1.0
    delayed_theta: float = 0.0
    min_budget_pct: float = 0.0
    max_budget_pct: float = 1.0


def _apply_adstock(spend: np.ndarray, cr: ChannelResponse) -> np.ndarray:
    """Apply adstock transformation based on channel config."""
    if cr.adstock_type == "geometric":
        return geometric_adstock_np(spend, cr.adstock_alpha, cr.adstock_max_lag)
    elif cr.adstock_type in ("weibull_cdf", "weibull_pdf"):
        return weibull_adstock_np(
            spend, cr.weibull_shape, cr.weibull_scale,
            cr.adstock_max_lag, cr.adstock_type.split("_")[1],
        )
    elif cr.adstock_type == "delayed":
        return delayed_adstock_np(spend, cr.adstock_alpha, cr.delayed_theta, cr.adstock_max_lag)
    return spend


def _apply_saturation(x: np.ndarray, cr: ChannelResponse) -> np.ndarray:
    """Apply saturation transformation based on channel config."""
    if cr.saturation_type == "hill":
        return hill_saturation_np(x, cr.saturation_lam, cr.saturation_k)
    elif cr.saturation_type == "logistic":
        return logistic_saturation_np(x, cr.saturation_lam, cr.saturation_k)
    elif cr.saturation_type == "s_curve":
        return gompertz_saturation_np(x, a=1.0, b=cr.saturation_lam, c=cr.saturation_k)
    return x


def _channel_response(spend_level: float, cr: ChannelResponse, n_periods: int = 52) -> float:
    """
    Compute total expected response for a constant weekly spend level over n_periods.
    """
    spend_series = np.full(n_periods, spend_level)
    adstocked = _apply_adstock(spend_series, cr)
    saturated = _apply_saturation(adstocked, cr)
    return float(cr.beta * saturated.sum())


def extract_posterior_params(model_instance) -> list[ChannelResponse]:
    """
    Extract posterior mean parameters from a fitted BayesianMMM
    to build ChannelResponse objects for optimization.
    """
    if model_instance.trace is None:
        raise RuntimeError("Model not fitted.")

    posterior = model_instance.trace.posterior
    cfg = model_instance.config
    responses = []

    for ch_cfg in cfg.channels:
        params = {
            "name": ch_cfg.name,
            "adstock_type": ch_cfg.adstock_type,
            "adstock_max_lag": ch_cfg.adstock_max_lag,
            "saturation_type": ch_cfg.saturation_type,
            "min_budget_pct": ch_cfg.min_budget_pct,
            "max_budget_pct": ch_cfg.max_budget_pct,
        }

        # Adstock params
        if ch_cfg.adstock_type == "geometric":
            params["adstock_alpha"] = float(posterior[f"adstock_alpha_{ch_cfg.name}"].mean())
        elif ch_cfg.adstock_type in ("weibull_cdf", "weibull_pdf"):
            params["adstock_alpha"] = 0.5
            params["weibull_shape"] = float(posterior[f"weibull_shape_{ch_cfg.name}"].mean())
            params["weibull_scale"] = float(posterior[f"weibull_scale_{ch_cfg.name}"].mean())
        elif ch_cfg.adstock_type == "delayed":
            params["adstock_alpha"] = float(posterior[f"adstock_alpha_{ch_cfg.name}"].mean())
            params["delayed_theta"] = float(posterior[f"adstock_theta_{ch_cfg.name}"].mean())
        else:
            params["adstock_alpha"] = 0.0

        # Saturation params
        if ch_cfg.saturation_type in ("hill", "logistic"):
            params["saturation_lam"] = float(posterior[f"sat_lam_{ch_cfg.name}"].mean())
            params["saturation_k"] = float(posterior[f"sat_k_{ch_cfg.name}"].mean())
        else:
            params["saturation_lam"] = 0.5
            params["saturation_k"] = 1.0

        params["beta"] = float(posterior[f"beta_{ch_cfg.name}"].mean())
        responses.append(ChannelResponse(**params))

    return responses


class BudgetOptimizer:
    """
    Optimal budget allocation engine.

    Parameters
    ----------
    channel_responses : List of ChannelResponse with learned parameters.
    total_budget : Total budget to allocate (in same units as spend data).
    n_periods : Planning horizon in periods (e.g. 52 for annual).
    """

    def __init__(
        self,
        channel_responses: list[ChannelResponse],
        total_budget: float,
        n_periods: int = 52,
    ):
        self.channels = channel_responses
        self.total_budget = total_budget
        self.n_periods = n_periods
        self.n_channels = len(channel_responses)
        self._optimal_allocation: np.ndarray | None = None

    def _objective(self, allocation: np.ndarray) -> float:
        """Negative total response (minimize → maximize response)."""
        total = 0.0
        for i, cr in enumerate(self.channels):
            weekly_spend = allocation[i] / self.n_periods
            total += _channel_response(weekly_spend, cr, self.n_periods)
        return -total

    def optimize(self) -> pd.DataFrame:
        """
        Find optimal budget allocation using SLSQP.

        Returns
        -------
        DataFrame with columns: channel, optimal_budget, optimal_pct,
        expected_contribution, roi.
        """
        # Constraints: allocations sum to total_budget
        constraints = [{"type": "eq", "fun": lambda x: x.sum() - self.total_budget}]

        # Bounds per channel
        bounds = [
            (
                self.total_budget * cr.min_budget_pct,
                self.total_budget * cr.max_budget_pct,
            )
            for cr in self.channels
        ]

        # Initial: equal allocation
        x0 = np.full(self.n_channels, self.total_budget / self.n_channels)

        result = minimize(
            self._objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning("Optimizer did not converge: %s", result.message)

        self._optimal_allocation = result.x

        records = []
        for i, cr in enumerate(self.channels):
            budget = result.x[i]
            weekly = budget / self.n_periods
            contribution = _channel_response(weekly, cr, self.n_periods)
            records.append({
                "channel": cr.name,
                "optimal_budget": budget,
                "optimal_pct": budget / self.total_budget * 100,
                "expected_contribution": contribution,
                "roi": contribution / (budget + 1e-12),
            })

        df = pd.DataFrame(records)
        logger.info("Optimization complete. Top channel: %s (%.1f%%)",
                     df.iloc[df["optimal_pct"].argmax()]["channel"],
                     df["optimal_pct"].max())
        return df

    def scenario_analysis(
        self,
        budget_multipliers: list[float] = [0.7, 0.85, 1.0, 1.15, 1.3],
    ) -> pd.DataFrame:
        """
        Run optimization at different total budget levels.

        Parameters
        ----------
        budget_multipliers : Fractions of the base total_budget to test.

        Returns
        -------
        DataFrame with scenario results.
        """
        records = []
        for mult in budget_multipliers:
            scenario_budget = self.total_budget * mult
            opt = BudgetOptimizer(self.channels, scenario_budget, self.n_periods)
            alloc_df = opt.optimize()
            total_contribution = alloc_df["expected_contribution"].sum()
            records.append({
                "budget_multiplier": mult,
                "total_budget": scenario_budget,
                "total_expected_contribution": total_contribution,
                "marginal_roi": total_contribution / (scenario_budget + 1e-12),
            })

        return pd.DataFrame(records)

    def response_curves(self, n_points: int = 100) -> pd.DataFrame:
        """
        Generate per-channel response curves for visualization.

        Returns
        -------
        DataFrame with columns: channel, spend_level, response.
        """
        records = []
        for cr in self.channels:
            max_spend = self.total_budget * cr.max_budget_pct / self.n_periods * 2
            spend_levels = np.linspace(0, max_spend, n_points)
            for s in spend_levels:
                resp = _channel_response(s, cr, self.n_periods)
                records.append({
                    "channel": cr.name,
                    "weekly_spend": s,
                    "annual_spend": s * self.n_periods,
                    "total_response": resp,
                })
        return pd.DataFrame(records)
