"""
Adstock (carry-over / decay) transformations for media variables.

Implements the core adstock functions used in Marketing Mix Models to
capture the lagged effect of advertising exposure on sales. Supports
both NumPy (for data prep) and PyTensor (for Bayesian model graphs).

Supported types:
    - Geometric decay:  x_t' = x_t + α * x_{t-1}'
    - Weibull CDF:      weights from Weibull CDF (flexible shape)
    - Weibull PDF:      weights from Weibull PDF (peaked then decay)
    - Delayed adstock:  peak effect after θ periods, then geometric decay
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt


# ===================================================================
# NumPy implementations (for preprocessing / plotting)
# ===================================================================

def geometric_adstock_np(
    x: np.ndarray,
    alpha: float,
    max_lag: int = 12,
    normalize: bool = True,
) -> np.ndarray:
    """
    Geometric (exponential) decay adstock.

    Parameters
    ----------
    x : 1-D array of spend/impressions per period.
    alpha : Retention rate in [0, 1). Higher = longer carry-over.
    max_lag : Maximum lag periods to consider.
    normalize : If True, normalize the decay weights to sum to 1.

    Returns
    -------
    Adstocked array of same length as x.
    """
    weights = np.array([alpha ** i for i in range(max_lag)])
    if normalize:
        weights /= weights.sum()

    padded = np.concatenate([np.zeros(max_lag - 1), x])
    out = np.convolve(padded, weights, mode="valid")[: len(x)]
    return out


def weibull_adstock_np(
    x: np.ndarray,
    shape: float,
    scale: float,
    max_lag: int = 12,
    adstock_type: str = "cdf",
    normalize: bool = True,
) -> np.ndarray:
    """
    Weibull-based adstock (CDF or PDF parameterization).

    The Weibull distribution provides flexible decay shapes:
    - shape < 1: rapid initial decay
    - shape = 1: equivalent to geometric (exponential)
    - shape > 1: delayed peak then decay

    Parameters
    ----------
    x : 1-D spend/impressions array.
    shape : Weibull shape parameter (k > 0).
    scale : Weibull scale parameter (λ > 0).
    max_lag : Maximum lag.
    adstock_type : 'cdf' for survival-function weights, 'pdf' for density weights.
    normalize : Normalize weights to sum to 1.
    """
    lags = np.arange(max_lag) + 1e-6

    if adstock_type == "cdf":
        weights = 1.0 - (1.0 - np.exp(-((lags / scale) ** shape)))
    else:
        weights = (
            (shape / scale)
            * (lags / scale) ** (shape - 1)
            * np.exp(-((lags / scale) ** shape))
        )

    if normalize:
        weights /= weights.sum() + 1e-12

    padded = np.concatenate([np.zeros(max_lag - 1), x])
    out = np.convolve(padded, weights, mode="valid")[: len(x)]
    return out


def delayed_adstock_np(
    x: np.ndarray,
    alpha: float,
    theta: float,
    max_lag: int = 12,
    normalize: bool = True,
) -> np.ndarray:
    """
    Delayed adstock: peak effect at lag=θ, then geometric decay.

    Weights ∝ α^{(lag - θ)²} which peaks at θ and decays symmetrically.
    This captures channels where the effect is not immediate (e.g., TV
    brand campaigns that take time to influence purchase decisions).
    """
    lags = np.arange(max_lag).astype(float)
    weights = alpha ** ((lags - theta) ** 2)
    if normalize:
        weights /= weights.sum() + 1e-12

    padded = np.concatenate([np.zeros(max_lag - 1), x])
    out = np.convolve(padded, weights, mode="valid")[: len(x)]
    return out


# ===================================================================
# PyTensor implementations (for Bayesian model graph)
# ===================================================================

def geometric_adstock_pt(
    x: pt.TensorVariable,
    alpha: pt.TensorVariable,
    max_lag: int = 12,
    normalize: bool = True,
) -> pt.TensorVariable:
    """
    Geometric adstock as a PyTensor scan operation.

    Used inside the PyMC model to allow gradient-based sampling
    over the adstock parameter α.
    """
    def _step(x_t, carry, alpha):
        return x_t + alpha * carry

    result, _ = pt.scan(
        fn=_step,
        sequences=[x],
        outputs_info=[pt.zeros(())],
        non_sequences=[alpha],
    )
    return result


def weibull_adstock_pt(
    x: pt.TensorVariable,
    shape: pt.TensorVariable,
    scale: pt.TensorVariable,
    max_lag: int = 12,
    adstock_type: str = "cdf",
) -> pt.TensorVariable:
    """
    Weibull adstock in PyTensor via convolution with learnable weights.
    """
    lags = pt.arange(max_lag).astype("float64") + 1e-6

    if adstock_type == "cdf":
        weights = pt.exp(-((lags / scale) ** shape))
    else:
        weights = (
            (shape / scale)
            * (lags / scale) ** (shape - 1)
            * pt.exp(-((lags / scale) ** shape))
        )

    weights = weights / (weights.sum() + 1e-12)
    weights = weights[::-1]

    x_padded = pt.concatenate([pt.zeros(max_lag - 1), x])

    # 1D convolution via scan
    n = x.shape[0]

    def _conv_step(i, x_pad, w, ml):
        return pt.dot(x_pad[i: i + ml], w)

    result, _ = pt.scan(
        fn=_conv_step,
        sequences=[pt.arange(n)],
        non_sequences=[x_padded, weights, max_lag],
    )
    return result


def delayed_adstock_pt(
    x: pt.TensorVariable,
    alpha: pt.TensorVariable,
    theta: pt.TensorVariable,
    max_lag: int = 12,
) -> pt.TensorVariable:
    """Delayed adstock in PyTensor."""
    lags = pt.arange(max_lag).astype("float64")
    weights = alpha ** ((lags - theta) ** 2)
    weights = weights / (weights.sum() + 1e-12)
    weights = weights[::-1]

    x_padded = pt.concatenate([pt.zeros(max_lag - 1), x])
    n = x.shape[0]

    def _conv_step(i, x_pad, w, ml):
        return pt.dot(x_pad[i: i + ml], w)

    result, _ = pt.scan(
        fn=_conv_step,
        sequences=[pt.arange(n)],
        non_sequences=[x_padded, weights, max_lag],
    )
    return result
