"""Risk metrics and risk management tools."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Literal, Optional, Tuple
from scipy.optimize import minimize


class RiskMetrics:
    """
    Calculate various risk metrics for portfolios.
    
    Parameters:
    -----------
    returns : DataFrame or Series
        Asset or portfolio returns
    """
    
    def __init__(self, returns: Union[pd.DataFrame, pd.Series]):
        self.returns = returns
    
    def var_historical(self, confidence: float = 0.95,
                      weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate historical Value at Risk.
        
        Parameters:
        -----------
        confidence : float
            Confidence level (e.g., 0.95 for 95%)
        weights : array, optional
            Portfolio weights (if returns is DataFrame)
        
        Returns:
        --------
        float : VaR (negative number)
        """
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        return np.percentile(port_returns, (1 - confidence) * 100)
    
    def var_parametric(self, confidence: float = 0.95,
                      weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate parametric (variance-covariance) VaR.
        
        Parameters:
        -----------
        confidence : float
            Confidence level
        weights : array, optional
            Portfolio weights
        
        Returns:
        --------
        float : VaR
        """
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        mean = np.mean(port_returns)
        std = np.std(port_returns, ddof=1)
        z_score = stats.norm.ppf(1 - confidence)
        
        return mean + z_score * std
    
    def var_monte_carlo(self, confidence: float = 0.95,
                       weights: Optional[np.ndarray] = None,
                       n_sims: int = 100000) -> float:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Parameters:
        -----------
        confidence : float
            Confidence level
        weights : array, optional
            Portfolio weights
        n_sims : int
            Number of simulations
        
        Returns:
        --------
        float : VaR
        """
        if isinstance(self.returns, pd.DataFrame):
            mean = self.returns.mean().values
            cov = self.returns.cov().values
            
            np.random.seed(42)
            sim_returns = np.random.multivariate_normal(mean, cov, n_sims)
            
            if weights is not None:
                port_returns = sim_returns @ weights
            else:
                port_returns = sim_returns.flatten()
        else:
            mean = np.mean(self.returns)
            std = np.std(self.returns, ddof=1)
            
            np.random.seed(42)
            port_returns = np.random.normal(mean, std, n_sims)
        
        return np.percentile(port_returns, (1 - confidence) * 100)
    
    def cvar(self, confidence: float = 0.95,
            weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Parameters:
        -----------
        confidence : float
            Confidence level
        weights : array, optional
            Portfolio weights
        
        Returns:
        --------
        float : CVaR
        """
        var = self.var_historical(confidence, weights)
        
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        return np.mean(port_returns[port_returns <= var])
    
    def maximum_drawdown(self, weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate maximum drawdown.
        
        Parameters:
        -----------
        weights : array, optional
            Portfolio weights
        
        Returns:
        --------
        float : Maximum drawdown (negative number)
        """
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        equity_curve = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        
        return np.min(drawdown)
    
    def drawdown_series(self, weights: Optional[np.ndarray] = None) -> pd.Series:
        """
        Get drawdown series over time.
        
        Parameters:
        -----------
        weights : array, optional
            Portfolio weights
        
        Returns:
        --------
        Series : Drawdown over time
        """
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        equity_curve = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        
        return pd.Series(drawdown, index=port_returns.index)
    
    def volatility(self, weights: Optional[np.ndarray] = None,
                   annualize: bool = True) -> float:
        """
        Calculate portfolio volatility.
        
        Parameters:
        -----------
        weights : array, optional
            Portfolio weights
        annualize : bool
            Annualize the result
        
        Returns:
        --------
        float : Volatility
        """
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        vol = np.std(port_returns, ddof=1)
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol
    
    def tracking_error(self, benchmark_returns: pd.Series,
                      weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate tracking error against benchmark.
        
        Parameters:
        -----------
        benchmark_returns : Series
            Benchmark returns
        weights : array, optional
            Portfolio weights
        
        Returns:
        --------
        float : Tracking error (annualized)
        """
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        active_returns = port_returns - benchmark_returns
        return np.std(active_returns, ddof=1) * np.sqrt(252)
    
    def beta(self, benchmark_returns: pd.Series,
            weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate portfolio beta.
        
        Parameters:
        -----------
        benchmark_returns : Series
            Benchmark returns
        weights : array, optional
            Portfolio weights
        
        Returns:
        --------
        float : Beta
        """
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        covariance = np.cov(port_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        
        if benchmark_variance == 0:
            return 0.0
        
        return covariance / benchmark_variance
    
    def tail_ratio(self, weights: Optional[np.ndarray] = None,
                  percentile: float = 0.05) -> float:
        """
        Calculate tail ratio (95th percentile / 5th percentile).
        
        Parameters:
        -----------
        weights : array, optional
            Portfolio weights
        percentile : float
            Tail percentile
        
        Returns:
        --------
        float : Tail ratio
        """
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        upper = np.percentile(port_returns, (1 - percentile) * 100)
        lower = np.percentile(port_returns, percentile * 100)
        
        if lower == 0:
            return np.inf
        
        return abs(upper / lower)
    
    def skewness(self, weights: Optional[np.ndarray] = None) -> float:
        """Calculate return skewness."""
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        return stats.skew(port_returns)
    
    def kurtosis(self, weights: Optional[np.ndarray] = None) -> float:
        """Calculate return kurtosis."""
        if isinstance(self.returns, pd.DataFrame) and weights is not None:
            port_returns = self.returns @ weights
        else:
            port_returns = self.returns
        
        return stats.kurtosis(port_returns)
    
    def summary(self, weights: Optional[np.ndarray] = None,
               benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate risk summary.
        
        Parameters:
        -----------
        weights : array, optional
            Portfolio weights
        benchmark_returns : Series, optional
            Benchmark returns for relative metrics
        
        Returns:
        --------
        DataFrame : Risk summary
        """
        metrics = {
            'Volatility (Ann %)': self.volatility(weights) * 100,
            'VaR 95% (%)': self.var_historical(0.95, weights) * 100,
            'CVaR 95% (%)': self.cvar(0.95, weights) * 100,
            'Max Drawdown (%)': self.maximum_drawdown(weights) * 100,
            'Skewness': self.skewness(weights),
            'Kurtosis': self.kurtosis(weights),
            'Tail Ratio': self.tail_ratio(weights)
        }
        
        if benchmark_returns is not None:
            metrics['Beta'] = self.beta(benchmark_returns, weights)
            metrics['Tracking Error (%)'] = self.tracking_error(benchmark_returns, weights) * 100
        
        return pd.DataFrame([metrics]).T.rename(columns={0: 'Value'})


class RiskParity:
    """
    Risk parity portfolio construction.
    
    Equal risk contribution from each asset.
    """
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.cov_matrix = returns.cov() * 252
        self.n_assets = len(returns.columns)
        self.asset_names = returns.columns.tolist()
    
    def get_weights(self, risk_budget: Optional[np.ndarray] = None,
                   max_weight: float = 1.0) -> np.ndarray:
        """
        Calculate risk parity weights.
        
        Parameters:
        -----------
        risk_budget : array, optional
            Target risk budget per asset (default: equal)
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        array : Portfolio weights
        """
        if risk_budget is None:
            risk_budget = np.ones(self.n_assets) / self.n_assets
        
        def risk_contribution(w):
            port_var = w @ self.cov_matrix @ w
            marginal_risk = self.cov_matrix @ w
            rc = w * marginal_risk / np.sqrt(port_var)
            return rc
        
        def objective(w):
            rc = risk_contribution(w)
            target_rc = risk_budget * np.sum(rc)
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weight)] * self.n_assets
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, bounds=bounds,
                         constraints=constraints, method='SLSQP')
        
        return result.x
    
    def get_risk_contributions(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Calculate risk contributions for given weights.
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        
        Returns:
        --------
        DataFrame : Risk contributions
        """
        port_var = weights @ self.cov_matrix @ weights
        marginal_risk = self.cov_matrix @ weights
        rc = weights * marginal_risk / np.sqrt(port_var)
        rc_pct = rc / np.sum(rc) * 100
        
        return pd.DataFrame({
            'Asset': self.asset_names,
            'Weight (%)': weights * 100,
            'Risk Contribution (%)': rc_pct
        })