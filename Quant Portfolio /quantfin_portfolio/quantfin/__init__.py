"""
QuantFin Pro - Advanced Quantitative Finance Library

A comprehensive library for quantitative finance research and trading.
"""

__version__ = "1.0.0"
__author__ = "Mahad Mir"

from quantfin.core.config import Config
from quantfin.core.utils import (
    annualize_returns,
    annualize_volatility,
    sharpe_ratio,
    maximum_drawdown,
    calmar_ratio,
    sortino_ratio
)

__all__ = [
    'Config',
    'annualize_returns',
    'annualize_volatility',
    'sharpe_ratio',
    'maximum_drawdown',
    'calmar_ratio',
    'sortino_ratio'
]