"""Portfolio backtesting framework."""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Container for backtest results."""
    returns: pd.Series
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    metrics: Dict[str, float]


class PortfolioBacktest:
    """
    Backtest portfolio strategies with rebalancing.
    
    Parameters:
    -----------
    returns : DataFrame
        Asset returns
    rebalance_freq : str
        Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
    transaction_cost : float
        Transaction cost as fraction (e.g., 0.001 = 10 bps)
    """
    
    def __init__(self, returns: pd.DataFrame,
                 rebalance_freq: str = 'M',
                 transaction_cost: float = 0.001):
        self.returns = returns
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
    
    def run(self, weight_func: Callable,
           initial_capital: float = 100000.0,
           lookback_window: Optional[int] = None) -> BacktestResult:
        """
        Run portfolio backtest.
        
        Parameters:
        -----------
        weight_func : callable
            Function that takes (returns_window) and returns weights
        initial_capital : float
            Starting capital
        lookback_window : int, optional
            Lookback period for weight calculation
        
        Returns:
        --------
        BacktestResult : Backtest results
        """
        # Generate rebalancing dates
        if self.rebalance_freq == 'D':
            rebalance_dates = self.returns.index
        else:
            rebalance_dates = self.returns.resample(self.rebalance_freq).last().index
        
        # Initialize
        capital = initial_capital
        current_weights = pd.Series(0, index=self.returns.columns)
        weights_history = []
        portfolio_returns = []
        turnover_series = []
        
        for i, date in enumerate(self.returns.index):
            # Check if rebalancing day
            if date in rebalance_dates or i == 0:
                # Calculate lookback window
                if lookback_window and i >= lookback_window:
                    lookback = self.returns.iloc[i-lookback_window:i]
                else:
                    lookback = self.returns.iloc[:i] if i > 0 else self.returns.iloc[:1]
                
                # Get new weights
                try:
                    new_weights = weight_func(lookback)
                    new_weights = pd.Series(new_weights, index=self.returns.columns)
                    new_weights = new_weights.fillna(0)
                    new_weights = new_weights / new_weights.sum() if new_weights.sum() > 0 else new_weights
                except Exception as e:
                    # Fallback to equal weights
                    new_weights = pd.Series(1/len(self.returns.columns), 
                                          index=self.returns.columns)
                
                # Calculate turnover
                turnover = np.sum(np.abs(new_weights - current_weights)) / 2
                turnover_series.append(turnover)
                
                # Apply transaction costs
                cost = turnover * self.transaction_cost
                capital *= (1 - cost)
                
                current_weights = new_weights
            else:
                turnover_series.append(0)
            
            # Calculate portfolio return
            daily_return = self.returns.loc[date] @ current_weights
            portfolio_returns.append(daily_return)
            
            # Update capital
            capital *= (1 + daily_return)
            
            # Record weights
            weights_history.append(current_weights.copy())
        
        # Create results
        port_returns = pd.Series(portfolio_returns, index=self.returns.index)
        equity_curve = initial_capital * (1 + port_returns).cumprod()
        weights_df = pd.DataFrame(weights_history, index=self.returns.index)
        turnover_series = pd.Series(turnover_series, index=self.returns.index)
        
        # Calculate metrics
        metrics = self._calculate_metrics(port_returns, turnover_series)
        
        return BacktestResult(
            returns=port_returns,
            equity_curve=equity_curve,
            weights_history=weights_df,
            turnover=turnover_series,
            metrics=metrics
        )
    
    def _calculate_metrics(self, returns: pd.Series, 
                          turnover: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        from ..core.utils import sharpe_ratio, maximum_drawdown, calmar_ratio
        
        return {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': sharpe_ratio(returns),
            'max_drawdown': maximum_drawdown((1 + returns).cumprod()),
            'calmar_ratio': calmar_ratio(returns),
            'avg_turnover': turnover.mean(),
            'annual_turnover': turnover.sum() * (252 / len(turnover))
        }


class StrategyBacktest:
    """
    Backtest with predefined strategies.
    """
    
    def __init__(self, returns: pd.DataFrame,
                 rebalance_freq: str = 'M',
                 transaction_cost: float = 0.001):
        self.returns = returns
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.backtest = PortfolioBacktest(returns, rebalance_freq, transaction_cost)
    
    def equal_weight(self, initial_capital: float = 100000.0) -> BacktestResult:
        """Equal weight strategy."""
        def weight_func(lookback):
            n = len(self.returns.columns)
            return np.ones(n) / n
        
        return self.backtest.run(weight_func, initial_capital)
    
    def inverse_volatility(self, window: int = 63,
                          initial_capital: float = 100000.0) -> BacktestResult:
        """Inverse volatility weighting."""
        def weight_func(lookback):
            vols = lookback.iloc[-window:].std() * np.sqrt(252)
            inv_vols = 1 / (vols + 1e-6)
            return inv_vols / inv_vols.sum()
        
        return self.backtest.run(weight_func, initial_capital, window)
    
    def momentum(self, lookback: int = 126,
                initial_capital: float = 100000.0) -> BacktestResult:
        """Momentum strategy (top performers)."""
        def weight_func(lookback_window):
            # Calculate momentum (past returns)
            momentum = (1 + lookback_window.iloc[-lookback:]).prod() - 1
            
            # Select top half
            n_select = max(1, len(momentum) // 2)
            top_assets = momentum.nlargest(n_select).index
            
            weights = pd.Series(0, index=self.returns.columns)
            weights[top_assets] = 1 / n_select
            return weights.values
        
        return self.backtest.run(weight_func, initial_capital, lookback * 2)
    
    def minimum_variance(self, window: int = 63,
                        initial_capital: float = 100000.0) -> BacktestResult:
        """Minimum variance optimization."""
        from .optimization import PortfolioOptimizer
        
        def weight_func(lookback):
            try:
                optimizer = PortfolioOptimizer(lookback.iloc[-window:])
                weights, _ = optimizer.optimize_min_variance()
                return weights
            except:
                n = len(self.returns.columns)
                return np.ones(n) / n
        
        return self.backtest.run(weight_func, initial_capital, window * 2)
    
    def risk_parity(self, window: int = 63,
                   initial_capital: float = 100000.0) -> BacktestResult:
        """Risk parity strategy."""
        from .risk import RiskParity
        
        def weight_func(lookback):
            try:
                rp = RiskParity(lookback.iloc[-window:])
                return rp.get_weights()
            except:
                n = len(self.returns.columns)
                return np.ones(n) / n
        
        return self.backtest.run(weight_func, initial_capital, window * 2)


class BenchmarkComparison:
    """
    Compare strategy against benchmarks.
    """
    
    def __init__(self, strategy_result: BacktestResult,
                 benchmark_returns: pd.Series,
                 risk_free_rate: float = 0.045):
        self.strategy = strategy_result
        self.benchmark = benchmark_returns
        self.risk_free_rate = risk_free_rate
    
    def comparison_table(self) -> pd.DataFrame:
        """Generate comparison table."""
        from ..core.utils import (
            sharpe_ratio, maximum_drawdown, calmar_ratio,
            information_ratio, tracking_error, beta, alpha
        )
        
        # Align dates
        common_dates = self.strategy.returns.index.intersection(self.benchmark.index)
        strat_ret = self.strategy.returns.loc[common_dates]
        bench_ret = self.benchmark.loc[common_dates]
        
        metrics = {
            'Strategy': {
                'Total Return (%)': (1 + strat_ret).prod() - 1,
                'Annual Return (%)': strat_ret.mean() * 252,
                'Annual Vol (%)': strat_ret.std() * np.sqrt(252),
                'Sharpe Ratio': sharpe_ratio(strat_ret, self.risk_free_rate),
                'Max Drawdown (%)': maximum_drawdown((1 + strat_ret).cumprod()),
                'Calmar Ratio': calmar_ratio(strat_ret)
            },
            'Benchmark': {
                'Total Return (%)': (1 + bench_ret).prod() - 1,
                'Annual Return (%)': bench_ret.mean() * 252,
                'Annual Vol (%)': bench_ret.std() * np.sqrt(252),
                'Sharpe Ratio': sharpe_ratio(bench_ret, self.risk_free_rate),
                'Max Drawdown (%)': maximum_drawdown((1 + bench_ret).cumprod()),
                'Calmar Ratio': calmar_ratio(bench_ret)
            }
        }
        
        # Add relative metrics
        metrics['Relative'] = {
            'Total Return (%)': metrics['Strategy']['Total Return (%)'] - metrics['Benchmark']['Total Return (%)'],
            'Annual Return (%)': metrics['Strategy']['Annual Return (%)'] - metrics['Benchmark']['Annual Return (%)'],
            'Annual Vol (%)': tracking_error(strat_ret, bench_ret),
            'Sharpe Ratio': 0,
            'Max Drawdown (%)': 0,
            'Calmar Ratio': 0,
            'Information Ratio': information_ratio(strat_ret, bench_ret),
            'Beta': beta(strat_ret, bench_ret),
            'Alpha (%)': alpha(strat_ret, bench_ret, self.risk_free_rate)
        }
        
        return pd.DataFrame(metrics).round(4)
    
    def rolling_performance(self, window: int = 63) -> pd.DataFrame:
        """Calculate rolling performance comparison."""
        common_dates = self.strategy.returns.index.intersection(self.benchmark.index)
        strat_ret = self.strategy.returns.loc[common_dates]
        bench_ret = self.benchmark.loc[common_dates]
        
        results = []
        for i in range(window, len(strat_ret)):
            date = strat_ret.index[i]
            
            strat_window = strat_ret.iloc[i-window:i]
            bench_window = bench_ret.iloc[i-window:i]
            
            results.append({
                'date': date,
                'strategy_return': strat_window.mean() * 252,
                'benchmark_return': bench_window.mean() * 252,
                'strategy_sharpe': strat_window.mean() / strat_window.std() * np.sqrt(252),
                'benchmark_sharpe': bench_window.mean() / bench_window.std() * np.sqrt(252)
            })
        
        return pd.DataFrame(results).set_index('date')