"""Utility functions for quantitative finance calculations."""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy import stats


def annualize_returns(returns: Union[np.ndarray, pd.Series], 
                      periods_per_year: int = 252) -> float:
    """
    Annualize returns from periodic returns.
    
    Parameters:
    -----------
    returns : array-like
        Periodic returns (daily, monthly, etc.)
    periods_per_year : int
        Number of periods in a year (252 for daily, 12 for monthly)
    
    Returns:
    --------
    float : Annualized return
    """
    returns = np.array(returns)
    return np.mean(returns) * periods_per_year


def annualize_volatility(returns: Union[np.ndarray, pd.Series],
                         periods_per_year: int = 252) -> float:
    """
    Annualize volatility from periodic returns.
    
    Parameters:
    -----------
    returns : array-like
        Periodic returns
    periods_per_year : int
        Number of periods in a year
    
    Returns:
    --------
    float : Annualized volatility
    """
    returns = np.array(returns)
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: Union[np.ndarray, pd.Series],
                 risk_free_rate: float = 0.045,
                 periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters:
    -----------
    returns : array-like
        Periodic returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods in a year
    
    Returns:
    --------
    float : Sharpe ratio
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns, ddof=1) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)


def sortino_ratio(returns: Union[np.ndarray, pd.Series],
                  risk_free_rate: float = 0.045,
                  periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Parameters:
    -----------
    returns : array-like
        Periodic returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods in a year
    
    Returns:
    --------
    float : Sortino ratio
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns, ddof=1) == 0:
        return 0.0
    
    downside_std = np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)
    return np.mean(excess_returns) * periods_per_year / downside_std


def maximum_drawdown(equity_curve: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Parameters:
    -----------
    equity_curve : array-like
        Cumulative portfolio value over time
    
    Returns:
    --------
    float : Maximum drawdown (negative number)
    """
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return np.min(drawdown)


def calmar_ratio(returns: Union[np.ndarray, pd.Series],
                 periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Parameters:
    -----------
    returns : array-like
        Periodic returns
    periods_per_year : int
        Number of periods in a year
    
    Returns:
    --------
    float : Calmar ratio
    """
    returns = np.array(returns)
    equity_curve = np.cumprod(1 + returns)
    ann_return = annualize_returns(returns, periods_per_year)
    max_dd = abs(maximum_drawdown(equity_curve))
    
    if max_dd == 0:
        return 0.0
    
    return ann_return / max_dd


def information_ratio(returns: Union[np.ndarray, pd.Series],
                     benchmark_returns: Union[np.ndarray, pd.Series],
                     periods_per_year: int = 252) -> float:
    """
    Calculate Information ratio (active return / tracking error).
    
    Parameters:
    -----------
    returns : array-like
        Strategy returns
    benchmark_returns : array-like
        Benchmark returns
    periods_per_year : int
        Number of periods in a year
    
    Returns:
    --------
    float : Information ratio
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    active_returns = returns - benchmark_returns
    
    if np.std(active_returns, ddof=1) == 0:
        return 0.0
    
    return np.mean(active_returns) / np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)


def tracking_error(returns: Union[np.ndarray, pd.Series],
                   benchmark_returns: Union[np.ndarray, pd.Series],
                   periods_per_year: int = 252) -> float:
    """
    Calculate tracking error (standard deviation of active returns).
    
    Parameters:
    -----------
    returns : array-like
        Strategy returns
    benchmark_returns : array-like
        Benchmark returns
    periods_per_year : int
        Number of periods in a year
    
    Returns:
    --------
    float : Tracking error
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    active_returns = returns - benchmark_returns
    
    return np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)


def beta(returns: Union[np.ndarray, pd.Series],
         benchmark_returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate beta (systematic risk).
    
    Parameters:
    -----------
    returns : array-like
        Strategy returns
    benchmark_returns : array-like
        Benchmark returns
    
    Returns:
    --------
    float : Beta
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns, ddof=1)
    
    if benchmark_variance == 0:
        return 0.0
    
    return covariance / benchmark_variance


def alpha(returns: Union[np.ndarray, pd.Series],
          benchmark_returns: Union[np.ndarray, pd.Series],
          risk_free_rate: float = 0.045,
          periods_per_year: int = 252) -> float:
    """
    Calculate Jensen's alpha.
    
    Parameters:
    -----------
    returns : array-like
        Strategy returns
    benchmark_returns : array-like
        Benchmark returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods in a year
    
    Returns:
    --------
    float : Alpha (annualized)
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    
    beta_value = beta(returns, benchmark_returns)
    ann_return = annualize_returns(returns, periods_per_year)
    ann_benchmark = annualize_returns(benchmark_returns, periods_per_year)
    
    return ann_return - (risk_free_rate + beta_value * (ann_benchmark - risk_free_rate))


def var_historical(returns: Union[np.ndarray, pd.Series],
                   confidence: float = 0.95) -> float:
    """
    Calculate historical Value at Risk.
    
    Parameters:
    -----------
    returns : array-like
        Periodic returns
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns:
    --------
    float : VaR (negative number)
    """
    returns = np.array(returns)
    return np.percentile(returns, (1 - confidence) * 100)


def var_parametric(returns: Union[np.ndarray, pd.Series],
                   confidence: float = 0.95) -> float:
    """
    Calculate parametric (variance-covariance) VaR.
    
    Parameters:
    -----------
    returns : array-like
        Periodic returns
    confidence : float
        Confidence level
    
    Returns:
    --------
    float : VaR
    """
    returns = np.array(returns)
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    z_score = stats.norm.ppf(1 - confidence)
    
    return mean + z_score * std


def cvar_historical(returns: Union[np.ndarray, pd.Series],
                    confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    Parameters:
    -----------
    returns : array-like
        Periodic returns
    confidence : float
        Confidence level
    
    Returns:
    --------
    float : CVaR
    """
    returns = np.array(returns)
    var = var_historical(returns, confidence)
    return np.mean(returns[returns <= var])


def make_positive_definite(matrix: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Ensure matrix is positive definite by adding small value to diagonal.
    
    Parameters:
    -----------
    matrix : ndarray
        Input matrix
    epsilon : float
        Small value to add
    
    Returns:
    --------
    ndarray : Positive definite matrix
    """
    return matrix + epsilon * np.eye(matrix.shape[0])


def winsorize(returns: Union[np.ndarray, pd.Series],
              limits: tuple = (0.05, 0.05)) -> np.ndarray:
    """
    Winsorize returns to reduce outlier impact.
    
    Parameters:
    -----------
    returns : array-like
        Input returns
    limits : tuple
        Lower and upper percentiles to clip
    
    Returns:
    --------
    ndarray : Winsorized returns
    """
    returns = np.array(returns)
    lower = np.percentile(returns, limits[0] * 100)
    upper = np.percentile(returns, (1 - limits[1]) * 100)
    return np.clip(returns, lower, upper)