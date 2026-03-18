"""Portfolio optimization methods."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple, List, Literal
import warnings


class PortfolioOptimizer:
    """
    Portfolio optimization using various methods.
    
    Parameters:
    -----------
    returns : DataFrame
        Historical returns data (assets as columns)
    """
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        self.n_assets = len(returns.columns)
        self.asset_names = returns.columns.tolist()
    
    def optimize_mean_variance(self, target_return: Optional[float] = None,
                               risk_aversion: float = 1.0,
                               max_weight: float = 1.0,
                               min_weight: float = 0.0,
                               allow_short: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Mean-variance optimization (Markowitz).
        
        Parameters:
        -----------
        target_return : float, optional
            Target portfolio return. If None, maximize Sharpe.
        risk_aversion : float
            Risk aversion parameter (higher = more risk-averse)
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
        allow_short : bool
            Allow short positions
        
        Returns:
        --------
        tuple : (weights, results_dict)
        """
        def objective(w):
            port_return = w @ self.mean_returns
            port_risk = np.sqrt(w @ self.cov_matrix @ w)
            
            if target_return is not None:
                # Minimize risk subject to target return
                return port_risk
            else:
                # Maximize utility: return - risk_aversion * risk^2
                return -(port_return - risk_aversion * port_risk**2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ self.mean_returns - target_return
            })
        
        # Bounds
        if allow_short:
            bounds = [(-max_weight, max_weight)] * self.n_assets
        else:
            bounds = [(min_weight, max_weight)] * self.n_assets
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, 
                         constraints=constraints, method='SLSQP')
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
        
        weights = result.x
        port_return = weights @ self.mean_returns
        port_risk = np.sqrt(weights @ self.cov_matrix @ weights)
        
        return weights, {
            'return': port_return,
            'risk': port_risk,
            'sharpe': port_return / port_risk if port_risk > 0 else 0,
            'success': result.success
        }
    
    def optimize_max_sharpe(self, risk_free_rate: float = 0.045,
                           max_weight: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        Maximize Sharpe ratio.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        tuple : (weights, results_dict)
        """
        excess_returns = self.mean_returns - risk_free_rate
        
        def neg_sharpe(w):
            port_return = w @ excess_returns
            port_risk = np.sqrt(w @ self.cov_matrix @ w)
            return -port_return / port_risk if port_risk > 0 else 0
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weight)] * self.n_assets
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(neg_sharpe, x0, bounds=bounds,
                         constraints=constraints, method='SLSQP')
        
        weights = result.x
        port_return = weights @ self.mean_returns
        port_risk = np.sqrt(weights @ self.cov_matrix @ weights)
        
        return weights, {
            'return': port_return,
            'risk': port_risk,
            'sharpe': (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0,
            'success': result.success
        }
    
    def optimize_min_variance(self, max_weight: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        Minimize portfolio variance.
        
        Parameters:
        -----------
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        tuple : (weights, results_dict)
        """
        def objective(w):
            return w @ self.cov_matrix @ w
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weight)] * self.n_assets
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, bounds=bounds,
                         constraints=constraints, method='SLSQP')
        
        weights = result.x
        port_return = weights @ self.mean_returns
        port_risk = np.sqrt(weights @ self.cov_matrix @ weights)
        
        return weights, {
            'return': port_return,
            'risk': port_risk,
            'sharpe': port_return / port_risk if port_risk > 0 else 0,
            'success': result.success
        }
    
    def optimize_cvar(self, confidence: float = 0.95,
                     target_return: Optional[float] = None,
                     max_weight: float = 1.0,
                     n_scenarios: int = 10000) -> Tuple[np.ndarray, dict]:
        """
        Optimize portfolio using CVaR (Conditional Value at Risk).
        
        Parameters:
        -----------
        confidence : float
            Confidence level for CVaR
        target_return : float, optional
            Target portfolio return
        max_weight : float
            Maximum weight per asset
        n_scenarios : int
            Number of Monte Carlo scenarios
        
        Returns:
        --------
        tuple : (weights, results_dict)
        """
        # Generate scenarios using historical returns
        np.random.seed(42)
        scenarios = np.random.multivariate_normal(
            self.mean_returns / 252,  # Daily means
            self.cov_matrix / 252,     # Daily cov
            n_scenarios
        )
        
        def cvar_objective(w):
            port_returns = scenarios @ w
            var = np.percentile(port_returns, (1 - confidence) * 100)
            cvar = -np.mean(port_returns[port_returns <= var])
            return cvar
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ self.mean_returns - target_return
            })
        
        bounds = [(0, max_weight)] * self.n_assets
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(cvar_objective, x0, bounds=bounds,
                         constraints=constraints, method='SLSQP')
        
        weights = result.x
        port_return = weights @ self.mean_returns
        port_risk = cvar_objective(weights)
        
        return weights, {
            'return': port_return,
            'cvar': port_risk,
            'success': result.success
        }
    
    def optimize_risk_parity(self, max_weight: float = 1.0,
                            risk_budget: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Risk parity optimization (equal risk contribution).
        
        Parameters:
        -----------
        max_weight : float
            Maximum weight per asset
        risk_budget : array, optional
            Target risk budget per asset (default: equal)
        
        Returns:
        --------
        tuple : (weights, results_dict)
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
        
        weights = result.x
        port_return = weights @ self.mean_returns
        port_risk = np.sqrt(weights @ self.cov_matrix @ weights)
        
        # Calculate risk contributions
        rc = risk_contribution(weights)
        
        return weights, {
            'return': port_return,
            'risk': port_risk,
            'risk_contributions': rc,
            'success': result.success
        }
    
    def efficient_frontier(self, n_points: int = 50,
                          max_weight: float = 1.0) -> pd.DataFrame:
        """
        Generate efficient frontier.
        
        Parameters:
        -----------
        n_points : int
            Number of points on frontier
        max_weight : float
            Maximum weight per asset
        
        Returns:
        --------
        DataFrame : Efficient frontier points
        """
        # Find min and max returns
        min_return = np.min(self.mean_returns)
        max_return = np.max(self.mean_returns)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        results = []
        for target in target_returns:
            weights, info = self.optimize_mean_variance(
                target_return=target,
                max_weight=max_weight
            )
            results.append({
                'target_return': target,
                'return': info['return'],
                'risk': info['risk'],
                'sharpe': info['sharpe']
            })
        
        return pd.DataFrame(results)
    
    def get_weights_df(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Convert weights array to DataFrame.
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        
        Returns:
        --------
        DataFrame : Weights with asset names
        """
        return pd.DataFrame({
            'asset': self.asset_names,
            'weight': weights,
            'weight_pct': weights * 100
        }).sort_values('weight', ascending=False)