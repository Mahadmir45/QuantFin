"""
Utility functions used across the MMM Bayesian framework.

Includes scaling, date feature engineering, Fourier terms, and
general-purpose helpers for data manipulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_channel_data(
    df: pd.DataFrame,
    channel_cols: list[str],
    method: str = "max_abs",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Scale channel spend columns and return scaling factors.

    Parameters
    ----------
    df : DataFrame with raw channel spend.
    channel_cols : Column names to scale.
    method : 'max_abs' (default) or 'standard'.

    Returns
    -------
    Tuple of (scaled DataFrame copy, dict of {col: scale_factor}).
    """
    df_out = df.copy()
    factors: dict[str, float] = {}

    for col in channel_cols:
        if method == "max_abs":
            scaler = MaxAbsScaler()
        else:
            scaler = StandardScaler()
        vals = df_out[col].values.reshape(-1, 1)
        df_out[col] = scaler.fit_transform(vals).flatten()
        factors[col] = float(scaler.scale_[0]) if hasattr(scaler, "scale_") else 1.0

    return df_out, factors


# ---------------------------------------------------------------------------
# Date features
# ---------------------------------------------------------------------------

def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar features: year, month, week_of_year, day_of_week, quarter."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_week"] = dt.dt.dayofweek
    df["quarter"] = dt.dt.quarter
    return df


def fourier_features(
    dates: pd.Series,
    period: float,
    order: int,
    prefix: str = "fourier",
) -> pd.DataFrame:
    """
    Generate sine/cosine Fourier features for seasonality modeling.

    Parameters
    ----------
    dates : Series of datetime-like values.
    period : Period length in the same unit as the date index (e.g. 365.25 for yearly).
    order : Number of Fourier pairs.
    prefix : Column name prefix.

    Returns
    -------
    DataFrame with 2*order columns of sin/cos terms.
    """
    t = (pd.to_datetime(dates) - pd.Timestamp("2000-01-01")).dt.days.values
    cols: dict[str, np.ndarray] = {}
    for k in range(1, order + 1):
        cols[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=dates.index)


# ---------------------------------------------------------------------------
# Train / test split (time-based)
# ---------------------------------------------------------------------------

def time_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    holdout_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame into train/test by chronological order."""
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - holdout_pct))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAPE, RMSE, R², and NRMSE."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs(residuals / np.where(y_true == 0, 1, y_true)))) * 100
    nrmse = rmse / (y_true.max() - y_true.min()) if (y_true.max() - y_true.min()) > 0 else 0.0

    return {"MAPE": mape, "RMSE": rmse, "R2": r2, "NRMSE": nrmse}
