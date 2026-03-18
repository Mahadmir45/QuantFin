"""
Saturation (diminishing returns) curves for media response modeling.

These functions model the non-linear relationship between spend/exposure
and incremental response. At low spend levels the marginal return is high;
beyond a point of saturation, additional spend yields diminishing returns.

Supported types:
    - Hill function:     x^k / (λ^k + x^k)
    - Logistic function: 1 / (1 + exp(-k*(x - λ)))
    - Michaelis-Menten:  (Vmax * x) / (Km + x)  [special case of Hill with k=1]
    - S-curve (Gompertz): a * exp(-b * exp(-c * x))
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt


# ===================================================================
# NumPy implementations
# ===================================================================

def hill_saturation_np(
    x: np.ndarray,
    lam: float,
    k: float,
) -> np.ndarray:
    """
    Hill function saturation.

    Parameters
    ----------
    x : Adstocked spend (non-negative).
    lam : Half-saturation point (EC50). The spend level at which response = 50%.
    k : Steepness / Hill coefficient. Higher k = sharper transition.

    Returns
    -------
    Saturated response in [0, 1].
    """
    xk = np.power(np.maximum(x, 0.0), k)
    return xk / (np.power(lam, k) + xk + 1e-12)


def logistic_saturation_np(
    x: np.ndarray,
    lam: float,
    k: float = 1.0,
) -> np.ndarray:
    """
    Logistic (sigmoid) saturation centered at λ.

    Returns values in (0, 1), centered at 0.5 when x = λ.
    """
    return 1.0 / (1.0 + np.exp(-k * (x - lam)))


def michaelis_menten_np(
    x: np.ndarray,
    vmax: float,
    km: float,
) -> np.ndarray:
    """
    Michaelis-Menten saturation (Hill with k=1, scaled by Vmax).

    Commonly used in pharma-style MMM models. Returns values in [0, Vmax].
    """
    return (vmax * np.maximum(x, 0.0)) / (km + np.maximum(x, 0.0) + 1e-12)


def gompertz_saturation_np(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    """
    Gompertz S-curve saturation.

    This produces an asymmetric S-curve: slow start, rapid growth, then
    plateau. Useful for modeling channels with a minimum effective dose.
    """
    return a * np.exp(-b * np.exp(-c * np.maximum(x, 0.0)))


# ===================================================================
# PyTensor implementations (for Bayesian model graph)
# ===================================================================

def hill_saturation_pt(
    x: pt.TensorVariable,
    lam: pt.TensorVariable,
    k: pt.TensorVariable,
) -> pt.TensorVariable:
    """Hill saturation in PyTensor."""
    xk = pt.power(pt.maximum(x, 0.0), k)
    return xk / (pt.power(lam, k) + xk + 1e-12)


def logistic_saturation_pt(
    x: pt.TensorVariable,
    lam: pt.TensorVariable,
    k: pt.TensorVariable,
) -> pt.TensorVariable:
    """Logistic saturation in PyTensor."""
    return pt.sigmoid(k * (x - lam))


def michaelis_menten_pt(
    x: pt.TensorVariable,
    vmax: pt.TensorVariable,
    km: pt.TensorVariable,
) -> pt.TensorVariable:
    """Michaelis-Menten in PyTensor."""
    x_pos = pt.maximum(x, 0.0)
    return (vmax * x_pos) / (km + x_pos + 1e-12)


def gompertz_saturation_pt(
    x: pt.TensorVariable,
    a: pt.TensorVariable,
    b: pt.TensorVariable,
    c: pt.TensorVariable,
) -> pt.TensorVariable:
    """Gompertz S-curve in PyTensor."""
    return a * pt.exp(-b * pt.exp(-c * pt.maximum(x, 0.0)))
