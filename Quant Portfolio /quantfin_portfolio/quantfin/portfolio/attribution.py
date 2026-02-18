"""Performance attribution analysis."""

import numpy as np
import pandas as pd
from typing import Optional


class PerformanceAttribution:
    """
    Performance attribution for portfolios.
    
    Decomposes returns into allocation, selection, and interaction effects.
    
    Parameters:
    -----------
    port_returns : DataFrame
        Portfolio returns by asset
    port_weights : DataFrame
        Portfolio weights by asset
    bench_returns : DataFrame
        Benchmark returns by asset
    bench_weights : DataFrame
        Benchmark weights by asset
    """
    
    def __init__(self, port_returns: pd.DataFrame,
                 port_weights: pd.DataFrame,
                 bench_returns: pd.DataFrame,
                 bench_weights: pd.DataFrame):
        self.port_returns = port_returns
        self.port_weights = port_weights
        self.bench_returns = bench_returns
        self.bench_weights = bench_weights
    
    def brinson_attribution(self) -> pd.DataFrame:
        """
        Perform Brinson attribution analysis.
        
        Returns:
        --------
        DataFrame : Attribution results
        """
        results = []
        
        for date in self.port_returns.index:
            if date not in self.bench_returns.index:
                continue
            
            # Get data for this period
            port_ret = self.port_returns.loc[date]
            port_w = self.port_weights.loc[date]
            bench_ret = self.bench_returns.loc[date]
            bench_w = self.bench_weights.loc[date]
            
            # Calculate effects for each asset
            for asset in port_w.index:
                if asset not in bench_w.index:
                    continue
                
                w_p = port_w[asset]
                w_b = bench_w[asset]
                r_p = port_ret[asset]
                r_b = bench_ret[asset]
                
                # Brinson attribution
                allocation = (w_p - w_b) * r_b
                selection = w_b * (r_p - r_b)
                interaction = (w_p - w_b) * (r_p - r_b)
                
                results.append({
                    'date': date,
                    'asset': asset,
                    'allocation': allocation,
                    'selection': selection,
                    'interaction': interaction,
                    'total': allocation + selection + interaction
                })
        
        return pd.DataFrame(results)
    
    def summary(self) -> pd.DataFrame:
        """
        Generate attribution summary.
        
        Returns:
        --------
        DataFrame : Summary statistics
        """
        attr = self.brinson_attribution()
        
        summary = attr.groupby('asset')[['allocation', 'selection', 'interaction', 'total']].sum()
        summary.loc['Total'] = summary.sum()
        
        return summary
    
    def factor_attribution(self, factor_returns: pd.DataFrame,
                          factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """
        Perform factor-based attribution.
        
        Parameters:
        -----------
        factor_returns : DataFrame
            Factor returns over time
        factor_exposures : DataFrame
            Portfolio factor exposures
        
        Returns:
        --------
        DataFrame : Factor attribution
        """
        # Align dates
        common_dates = self.port_returns.index.intersection(factor_returns.index)
        
        port_ret = (self.port_returns @ self.port_weights.T).loc[common_dates].sum(axis=1)
        
        # Calculate factor contributions
        results = []
        for date in common_dates:
            for factor in factor_returns.columns:
                exposure = factor_exposures.loc[date, factor] if date in factor_exposures.index else 0
                factor_ret = factor_returns.loc[date, factor]
                contribution = exposure * factor_ret
                
                results.append({
                    'date': date,
                    'factor': factor,
                    'exposure': exposure,
                    'return': factor_ret,
                    'contribution': contribution
                })
        
        return pd.DataFrame(results)


class ReturnDecomposition:
    """
    Decompose portfolio returns into components.
    """
    
    @staticmethod
    def decompose_returns(portfolio_returns: pd.Series,
                         benchmark_returns: pd.Series,
                         risk_free_rate: float = 0.045) -> pd.DataFrame:
        """
        Decompose excess returns into components.
        
        Parameters:
        -----------
        portfolio_returns : Series
            Portfolio returns
        benchmark_returns : Series
            Benchmark returns
        risk_free_rate : float
            Risk-free rate
        
        Returns:
        --------
        DataFrame : Return decomposition
        """
        # Calculate metrics
        from ..core.utils import sharpe_ratio, information_ratio, beta, alpha
        
        port_sharpe = sharpe_ratio(portfolio_returns, risk_free_rate)
        bench_sharpe = sharpe_ratio(benchmark_returns, risk_free_rate)
        info_ratio = information_ratio(portfolio_returns, benchmark_returns)
        port_beta = beta(portfolio_returns, benchmark_returns)
        port_alpha = alpha(portfolio_returns, benchmark_returns, risk_free_rate)
        
        # Decomposition
        total_excess = portfolio_returns.mean() * 252 - risk_free_rate
        benchmark_return = benchmark_returns.mean() * 252 - risk_free_rate
        
        market_component = port_beta * benchmark_return
        alpha_component = port_alpha
        
        return pd.DataFrame({
            'Component': ['Total Excess Return', 'Market (Beta)', 'Alpha', 'Residual'],
            'Annual Return (%)': [
                total_excess * 100,
                market_component * 100,
                alpha_component * 100,
                (total_excess - market_component - alpha_component) * 100
            ],
            'Metric': ['', f'Beta: {port_beta:.2f}', f'IR: {info_ratio:.2f}', '']
        })


class RollingAttribution:
    """
    Rolling performance attribution over time.
    """
    
    def __init__(self, returns: pd.DataFrame, weights: pd.DataFrame,
                 window: int = 63):
        """
        Parameters:
        -----------
        returns : DataFrame
            Asset returns
        weights : DataFrame
            Portfolio weights
        window : int
            Rolling window (default: 63 days ~ 3 months)
        """
        self.returns = returns
        self.weights = weights
        self.window = window
    
    def rolling_metrics(self) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Returns:
        --------
        DataFrame : Rolling metrics over time
        """
        results = []
        
        for i in range(self.window, len(self.returns)):
            date = self.returns.index[i]
            
            # Get window data
            ret_window = self.returns.iloc[i-self.window:i]
            w_window = self.weights.iloc[i-self.window:i].mean()
            
            # Portfolio return
            port_ret = (ret_window @ w_window)
            
            # Calculate metrics
            total_return = (1 + port_ret).prod() - 1
            ann_return = total_return * (252 / self.window)
            ann_vol = port_ret.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Drawdown
            cumret = (1 + port_ret).cumprod()
            running_max = cumret.expanding().max()
            drawdown = (cumret - running_max) / running_max
            max_dd = drawdown.min()
            
            results.append({
                'date': date,
                'annual_return': ann_return,
                'annual_volatility': ann_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            })
        
        return pd.DataFrame(results).set_index('date')