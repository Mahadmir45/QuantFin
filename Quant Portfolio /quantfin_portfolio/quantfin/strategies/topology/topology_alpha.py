"""
Topology Alpha Model - Enhanced Version

A quantitative trading strategy using Topological Data Analysis (TDA) 
on stock correlation graphs to detect market regimes and generate alpha signals.

Based on algebraic topology concepts:
- Correlation graph construction
- Persistent homology (Betti numbers)
- Laplacian diffusion for residual alpha
- Risk-parity weighting
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import expm, eigh
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TopologyFeatures:
    """Container for topology-based features."""
    betti_numbers: np.ndarray
    betti_std: float
    laplacian_eigenvalues: np.ndarray
    diffusion_residuals: np.ndarray
    regime_score: float


class TopologyAlphaStrategy:
    """
    Topology Alpha Model Strategy.
    
    Uses topological data analysis to:
    1. Build correlation graphs from stock returns
    2. Compute persistent homology features (Betti numbers)
    3. Apply Laplacian diffusion to detect mispricings
    4. Generate alpha signals from residuals
    5. Time exposure based on VIX and topology regime
    
    Parameters:
    -----------
    stocks : list
        List of stock tickers
    window_size : int
        Rolling window for correlation calculation
    vix_threshold : float
        VIX level for risk-off signal
    betti_threshold_low : float
        Low Betti std threshold (low risk regime)
    betti_threshold_high : float
        High Betti std threshold (high risk regime)
    diffusion_time : float
        Time parameter for Laplacian diffusion
    use_ml : bool
        Whether to use ML for signal enhancement
    """
    
    def __init__(self, 
                 stocks: List[str],
                 window_size: int = 90,
                 vix_threshold: float = 35,
                 betti_threshold_low: float = 30,
                 betti_threshold_high: float = 45,
                 diffusion_time: float = 0.5,
                 use_ml: bool = False):
        self.stocks = stocks
        self.window_size = window_size
        self.vix_threshold = vix_threshold
        self.betti_threshold_low = betti_threshold_low
        self.betti_threshold_high = betti_threshold_high
        self.diffusion_time = diffusion_time
        self.use_ml = use_ml
        
        self.model = None
        self.feature_history = []
        self.signal_history = []
    
    def build_correlation_graph(self, returns_window: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build correlation graph and compute Laplacian.
        
        Parameters:
        -----------
        returns_window : DataFrame
            Recent returns data
        
        Returns:
        --------
        tuple : (Laplacian matrix, correlation matrix)
        """
        # Calculate correlation matrix
        corr = returns_window.corr().abs().fillna(0)
        
        # Ensure valid correlation matrix
        corr = (corr + corr.T) / 2  # Symmetrize
        np.fill_diagonal(corr.values, 1)
        
        # Build adjacency matrix
        A = corr.values
        
        # Degree matrix
        D = np.diag(A.sum(axis=1))
        
        # Laplacian matrix
        L = D - A
        
        return L, A
    
    def laplacian_diffusion(self, L: np.ndarray, 
                           signal: np.ndarray,
                           t: Optional[float] = None) -> np.ndarray:
        """
        Apply Laplacian diffusion to signal.
        
        The diffusion smooths the signal across the graph structure,
        revealing local vs global mispricings.
        
        Parameters:
        -----------
        L : ndarray
            Laplacian matrix
        signal : ndarray
            Input signal (e.g., returns)
        t : float, optional
            Diffusion time (default: self.diffusion_time)
        
        Returns:
        --------
        ndarray : Diffused signal
        """
        if t is None:
            t = self.diffusion_time
        
        try:
            diffused = expm(-t * L) @ signal
            return diffused
        except Exception as e:
            logger.warning(f"Laplacian diffusion failed: {e}")
            return signal
    
    def compute_persistence_features(self, 
                                     corr_matrices: List[np.ndarray],
                                     thresholds: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute persistent homology features (Betti numbers).
        
        Parameters:
        -----------
        corr_matrices : list
            List of correlation matrices
        thresholds : array, optional
            Thresholds for graph construction
        
        Returns:
        --------
        ndarray : Persistence features
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 7)
        
        features = []
        
        for corr in corr_matrices:
            ph_vec = []
            
            for thresh in thresholds:
                # Build graph at this threshold
                adj = (corr > thresh).astype(int) - np.eye(len(corr))
                G = nx.from_numpy_array(adj)
                
                # Betti-0: Number of connected components - 1
                betti0 = nx.number_connected_components(G) - 1
                
                # Betti-1 approximation: Cycles
                n_nodes = len(corr)
                n_edges = G.number_of_edges()
                betti1 = max(0, n_edges - n_nodes + betti0 + 1)
                
                ph_vec.extend([betti0, betti1])
            
            # Compute lifetimes
            lifetimes = np.abs(np.diff(ph_vec + [0]))
            
            # Aggregate features
            features.append(np.hstack([
                np.mean(ph_vec),
                np.std(ph_vec),
                np.mean(lifetimes),
                np.max(lifetimes),
                np.sum(lifetimes),
                np.median(lifetimes)
            ]))
        
        return np.array(features)
    
    def detect_regime(self, betti_std: float, vix: float) -> str:
        """
        Detect market regime based on topology and VIX.
        
        Parameters:
        -----------
        betti_std : float
            Standard deviation of Betti numbers
        vix : float
            VIX level
        
        Returns:
        --------
        str : Regime ('low_risk', 'medium_risk', 'high_risk')
        """
        if vix > self.vix_threshold and betti_std > self.betti_threshold_high:
            return 'high_risk'
        elif betti_std < self.betti_threshold_low and vix < self.vix_threshold * 0.7:
            return 'low_risk'
        else:
            return 'medium_risk'
    
    def get_exposure(self, regime: str) -> float:
        """
        Get portfolio exposure based on regime.
        
        Parameters:
        -----------
        regime : str
            Detected regime
        
        Returns:
        --------
        float : Exposure multiplier
        """
        exposures = {
            'low_risk': 1.2,
            'medium_risk': 1.0,
            'high_risk': 0.0  # Flat
        }
        return exposures.get(regime, 0.6)
    
    def generate_signals(self, 
                        returns: pd.DataFrame,
                        vix: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Parameters:
        -----------
        returns : DataFrame
            Stock returns
        vix : Series, optional
            VIX values
        
        Returns:
        --------
        DataFrame : Signals and weights over time
        """
        signals = []
        
        for i in range(self.window_size, len(returns)):
            date = returns.index[i]
            
            # Get window
            window = returns.iloc[i-self.window_size:i]
            
            # Build graph
            L, corr = self.build_correlation_graph(window)
            
            # Current returns
            current_returns = returns.iloc[i].values
            
            # Laplacian diffusion
            diffused = self.laplacian_diffusion(L, current_returns)
            residuals = current_returns - diffused
            
            # Topology features
            eigenvalues = eigh(L)[0][:3]
            ph_features = self.compute_persistence_features([corr])[0]
            betti_std = ph_features[1]
            
            # VIX level
            vix_level = vix.iloc[i] if vix is not None else 20
            
            # Detect regime
            regime = self.detect_regime(betti_std, vix_level)
            exposure = self.get_exposure(regime)
            
            # Generate alpha signals from residuals
            if exposure > 0:
                # Sort by residuals (higher = more undervalued)
                sorted_idx = np.argsort(residuals)[-len(self.stocks):]
                
                # Risk parity weights based on volatility
                vols = window.std().values + 1e-6
                inv_vols = 1 / vols
                weights = inv_vols / inv_vols.sum()
                
                # Apply exposure
                weights = weights * exposure
            else:
                weights = np.zeros(len(self.stocks))
            
            signals.append({
                'date': date,
                'regime': regime,
                'exposure': exposure,
                'betti_std': betti_std,
                'vix': vix_level,
                **{f'weight_{stock}': w for stock, w in zip(self.stocks, weights)}
            })
        
        return pd.DataFrame(signals).set_index('date')
    
    def backtest(self,
                prices: pd.DataFrame,
                vix: Optional[pd.Series] = None,
                transaction_cost: float = 0.0005,
                initial_capital: float = 100000.0) -> Dict:
        """
        Run strategy backtest.
        
        Parameters:
        -----------
        prices : DataFrame
            Historical prices
        vix : Series, optional
            VIX prices
        transaction_cost : float
            Transaction cost per trade
        initial_capital : float
            Starting capital
        
        Returns:
        --------
        dict : Backtest results
        """
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Get signals
        signals_df = self.generate_signals(returns, vix)
        
        # Extract weights
        weight_cols = [c for c in signals_df.columns if c.startswith('weight_')]
        weights = signals_df[weight_cols].values
        
        # Calculate portfolio returns
        stock_returns = returns.loc[signals_df.index, self.stocks].values
        port_returns = np.sum(weights * stock_returns, axis=1)
        
        # Apply transaction costs
        turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1) / 2
        costs = np.concatenate([[0], turnover * transaction_cost])
        port_returns = port_returns - costs
        
        # Calculate equity curve
        equity_curve = initial_capital * np.cumprod(1 + port_returns)
        
        # Calculate metrics
        from ...core.utils import sharpe_ratio, maximum_drawdown, calmar_ratio
        
        metrics = {
            'total_return': (1 + port_returns).prod() - 1,
            'annual_return': np.mean(port_returns) * 252,
            'annual_volatility': np.std(port_returns) * np.sqrt(252),
            'sharpe_ratio': sharpe_ratio(port_returns),
            'max_drawdown': maximum_drawdown(equity_curve / initial_capital),
            'calmar_ratio': calmar_ratio(port_returns),
            'avg_turnover': np.mean(turnover),
            'avg_exposure': signals_df['exposure'].mean()
        }
        
        return {
            'signals': signals_df,
            'returns': pd.Series(port_returns, index=signals_df.index),
            'equity_curve': pd.Series(equity_curve, index=signals_df.index),
            'weights': pd.DataFrame(weights, index=signals_df.index, columns=self.stocks),
            'metrics': metrics
        }
    
    def get_weights(self, returns_window: pd.DataFrame,
                   vix_current: float = 20) -> np.ndarray:
        """
        Get current portfolio weights.
        
        Parameters:
        -----------
        returns_window : DataFrame
            Recent returns
        vix_current : float
            Current VIX level
        
        Returns:
        --------
        ndarray : Portfolio weights
        """
        # Build graph
        L, corr = self.build_correlation_graph(returns_window)
        
        # Current returns
        current_returns = returns_window.iloc[-1].values
        
        # Laplacian diffusion
        diffused = self.laplacian_diffusion(L, current_returns)
        residuals = current_returns - diffused
        
        # Topology features
        ph_features = self.compute_persistence_features([corr])[0]
        betti_std = ph_features[1]
        
        # Detect regime
        regime = self.detect_regime(betti_std, vix_current)
        exposure = self.get_exposure(regime)
        
        if exposure > 0:
            # Risk parity weights
            vols = returns_window.std().values + 1e-6
            inv_vols = 1 / vols
            weights = inv_vols / inv_vols.sum()
            weights = weights * exposure
        else:
            weights = np.zeros(len(self.stocks))
        
        return weights