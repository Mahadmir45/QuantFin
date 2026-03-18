"""Performance metrics for backtesting."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from scipy import stats


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
    -----------
    returns : Series
        Strategy returns
    benchmark_returns : Series, optional
        Benchmark returns
    risk_free_rate : float
        Annual risk-free rate
    """
    
    def __init__(self, returns: pd.Series,
                 benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.045):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
    
    def total_return(self) -> float:
        """Calculate total return."""
        return (1 + self.returns).prod() - 1
    
    def annualized_return(self) -> float:
        """Calculate annualized return."""
        return self.returns.mean() * 252
    
    def annualized_volatility(self) -> float:
        """Calculate annualized volatility."""
        return self.returns.std() * np.sqrt(252)
    
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = self.returns - self.risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        excess_returns = self.returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() * 252 / downside_std
    
    def maximum_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        equity_curve = (1 + self.returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        ann_return = self.annualized_return()
        max_dd = abs(self.maximum_drawdown())
        
        if max_dd == 0:
            return 0.0
        
        return ann_return / max_dd
    
    def var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(self.returns, (1 - confidence) * 100)
    
    def cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk."""
        var = self.var(confidence)
        return self.returns[self.returns <= var].mean()
    
    def beta(self) -> Optional[float]:
        """Calculate beta."""
        if self.benchmark_returns is None:
            return None
        
        aligned = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return None
        
        covariance = aligned.cov().iloc[0, 1]
        benchmark_variance = aligned.iloc[:, 1].var()
        
        if benchmark_variance == 0:
            return 0.0
        
        return covariance / benchmark_variance
    
    def alpha(self) -> Optional[float]:
        """Calculate Jensen's alpha."""
        if self.benchmark_returns is None:
            return None
        
        beta = self.beta()
        if beta is None:
            return None
        
        ann_return = self.annualized_return()
        bench_return = self.benchmark_returns.mean() * 252
        
        return ann_return - (self.risk_free_rate + beta * (bench_return - self.risk_free_rate))
    
    def information_ratio(self) -> Optional[float]:
        """Calculate Information ratio."""
        if self.benchmark_returns is None:
            return None
        
        aligned = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return None
        
        active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        
        if active_returns.std() == 0:
            return 0.0
        
        return active_returns.mean() / active_returns.std() * np.sqrt(252)
    
    def tracking_error(self) -> Optional[float]:
        """Calculate tracking error."""
        if self.benchmark_returns is None:
            return None
        
        aligned = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return None
        
        active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        return active_returns.std() * np.sqrt(252)
    
    def win_rate(self) -> float:
        """Calculate win rate."""
        return (self.returns > 0).mean()
    
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        gross_profit = self.returns[self.returns > 0].sum()
        gross_loss = abs(self.returns[self.returns < 0].sum())
        
        if gross_loss == 0:
            return np.inf
        
        return gross_profit / gross_loss
    
    def skewness(self) -> float:
        """Calculate return skewness."""
        return stats.skew(self.returns)
    
    def kurtosis(self) -> float:
        """Calculate return kurtosis."""
        return stats.kurtosis(self.returns)
    
    def summary(self) -> pd.DataFrame:
        """Generate performance summary."""
        metrics = {
            'Total Return (%)': self.total_return() * 100,
            'Annual Return (%)': self.annualized_return() * 100,
            'Annual Volatility (%)': self.annualized_volatility() * 100,
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown (%)': self.maximum_drawdown() * 100,
            'Calmar Ratio': self.calmar_ratio(),
            'VaR 95% (%)': self.var(0.95) * 100,
            'CVaR 95% (%)': self.cvar(0.95) * 100,
            'Win Rate (%)': self.win_rate() * 100,
            'Profit Factor': self.profit_factor(),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis()
        }
        
        if self.benchmark_returns is not None:
            metrics['Beta'] = self.beta()
            metrics['Alpha (%)'] = self.alpha() * 100 if self.alpha() else None
            metrics['Information Ratio'] = self.information_ratio()
            metrics['Tracking Error (%)'] = self.tracking_error() * 100
        
        return pd.DataFrame([metrics]).T.rename(columns={0: 'Value'})
    
    def monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns."""
        monthly = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        if self.benchmark_returns is not None:
            bench_monthly = self.benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly = pd.concat([monthly, bench_monthly], axis=1)
            monthly.columns = ['Strategy', 'Benchmark']
        
        return monthly
    
    def rolling_metrics(self, window: int = 63) -> pd.DataFrame:
        """Calculate rolling metrics."""
        results = []
        
        for i in range(window, len(self.returns)):
            window_returns = self.returns.iloc[i-window:i]
            
            metrics = PerformanceMetrics(window_returns, risk_free_rate=self.risk_free_rate)
            
            results.append({
                'date': self.returns.index[i],
                'return': metrics.annualized_return(),
                'volatility': metrics.annualized_volatility(),
                'sharpe': metrics.sharpe_ratio(),
                'max_dd': metrics.maximum_drawdown()
            })
        
        return pd.DataFrame(results).set_index('date')