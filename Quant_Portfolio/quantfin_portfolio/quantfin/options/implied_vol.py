"""Implied volatility calculation and surface construction."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, newton
from typing import List, Tuple, Optional, Literal
from scipy.interpolate import griddata, SmoothBivariateSpline
import matplotlib.pyplot as plt


class ImpliedVolatility:
    """
    Calculate implied volatility from market prices.
    
    Parameters:
    -----------
    S : float
        Current stock price
    r : float
        Risk-free rate
    q : float, optional
        Dividend yield
    """
    
    def __init__(self, S: float, r: float, q: float = 0.0):
        self.S = S
        self.r = r
        self.q = q
    
    def calculate(self, K: float, T: float, market_price: float,
                  option_type: Literal['call', 'put'] = 'call',
                  method: Literal['brent', 'newton'] = 'brent',
                  initial_guess: float = 0.2) -> float:
        """
        Calculate implied volatility for a single option.
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to expiration
        market_price : float
            Observed market price
        option_type : str
            'call' or 'put'
        method : str
            'brent' or 'newton'
        initial_guess : float
            Initial volatility guess
        
        Returns:
        --------
        float : Implied volatility
        """
        from .models.black_scholes import BlackScholes
        
        def objective(sigma):
            bs = BlackScholes(self.S, K, T, self.r, sigma, self.q)
            return bs.price(option_type) - market_price
        
        try:
            if method == 'brent':
                # Brent's method - more robust
                iv = brentq(objective, 1e-6, 5.0, xtol=1e-6)
            else:
                # Newton-Raphson - faster but needs good initial guess
                iv = newton(objective, initial_guess, tol=1e-6, maxiter=100)
            return iv
        except (ValueError, RuntimeError):
            return np.nan
    
    def calculate_batch(self, strikes: List[float], maturities: List[float],
                       market_prices: np.ndarray,
                       option_type: Literal['call', 'put'] = 'call') -> np.ndarray:
        """
        Calculate implied volatilities for multiple options.
        
        Parameters:
        -----------
        strikes : list
            Strike prices
        maturities : list
            Time to maturities
        market_prices : ndarray
            Matrix of market prices (strikes x maturities)
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        ndarray : Matrix of implied volatilities
        """
        iv_surface = np.zeros_like(market_prices)
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if not np.isnan(market_prices[i, j]):
                    iv_surface[i, j] = self.calculate(
                        K, T, market_prices[i, j], option_type
                    )
        
        return iv_surface


class IVSurface:
    """
    Implied volatility surface construction and analysis.
    
    Parameters:
    -----------
    strikes : array-like
        Strike prices
    maturities : array-like
        Time to maturities
    iv_data : ndarray
        Implied volatility matrix
    """
    
    def __init__(self, strikes: np.ndarray, maturities: np.ndarray, 
                 iv_data: np.ndarray):
        self.strikes = strikes
        self.maturities = maturities
        self.iv_data = iv_data
        self.S = None  # Will be set from context
        
        # Create interpolator
        self._create_interpolator()
    
    def _create_interpolator(self):
        """Create bivariate spline interpolator."""
        # Filter out NaN values
        valid = ~np.isnan(self.iv_data)
        if np.sum(valid) > 10:  # Need enough points
            K_mesh, T_mesh = np.meshgrid(self.strikes, self.maturities)
            self.interpolator = SmoothBivariateSpline(
                K_mesh[valid.T].flatten(),
                T_mesh[valid.T].flatten(),
                self.iv_data[valid.T].flatten()
            )
        else:
            self.interpolator = None
    
    def get_iv(self, K: float, T: float) -> float:
        """
        Get interpolated implied volatility.
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to maturity
        
        Returns:
        --------
        float : Implied volatility
        """
        if self.interpolator is not None:
            return max(self.interpolator.ev(K, T), 0)
        
        # Fallback to nearest neighbor
        k_idx = np.argmin(np.abs(self.strikes - K))
        t_idx = np.argmin(np.abs(self.maturities - T))
        return self.iv_data[k_idx, t_idx]
    
    def get_skew(self, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get volatility skew at a given maturity.
        
        Parameters:
        -----------
        T : float
            Time to maturity
        
        Returns:
        --------
        tuple : (strikes, ivs)
        """
        t_idx = np.argmin(np.abs(self.maturities - T))
        ivs = self.iv_data[:, t_idx]
        valid = ~np.isnan(ivs)
        return self.strikes[valid], ivs[valid]
    
    def get_term_structure(self, K: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get volatility term structure at a given strike.
        
        Parameters:
        -----------
        K : float
            Strike price
        
        Returns:
        --------
        tuple : (maturities, ivs)
        """
        k_idx = np.argmin(np.abs(self.strikes - K))
        ivs = self.iv_data[k_idx, :]
        valid = ~np.isnan(ivs)
        return self.maturities[valid], ivs[valid]
    
    def atm_vol(self, T: float) -> float:
        """
        Get at-the-money volatility.
        
        Parameters:
        -----------
        T : float
            Time to maturity
        
        Returns:
        --------
        float : ATM implied volatility
        """
        if self.S is None:
            # Use middle strike as proxy
            atm_strike = np.median(self.strikes)
        else:
            atm_strike = self.S
        
        return self.get_iv(atm_strike, T)
    
    def skew_slope(self, T: float) -> float:
        """
        Calculate skew slope (dIV/dK) at ATM.
        
        Parameters:
        -----------
        T : float
            Time to maturity
        
        Returns:
        --------
        float : Skew slope
        """
        if self.S is None:
            S = np.median(self.strikes)
        else:
            S = self.S
        
        # Calculate slope using finite differences
        dK = S * 0.01  # 1% of spot
        iv_up = self.get_iv(S + dK, T)
        iv_down = self.get_iv(S - dK, T)
        
        return (iv_up - iv_down) / (2 * dK)
    
    def plot_surface(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the implied volatility surface.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, ax : Matplotlib figure and axis
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        K_mesh, T_mesh = np.meshgrid(self.strikes, self.maturities * 365)
        
        # Mask NaN values
        iv_plot = np.where(np.isnan(self.iv_data.T), 0, self.iv_data.T)
        
        surf = ax.plot_surface(K_mesh, T_mesh, iv_plot * 100, 
                               cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Days to Expiration')
        ax.set_zlabel('Implied Volatility (%)')
        ax.set_title('Implied Volatility Surface')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return fig, ax
    
    def plot_skew(self, maturities: List[float] = None, 
                  figsize: Tuple[int, int] = (10, 6)):
        """
        Plot volatility skew for different maturities.
        
        Parameters:
        -----------
        maturities : list, optional
            List of maturities to plot
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, ax : Matplotlib figure and axis
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if maturities is None:
            maturities = [self.maturities[len(self.maturities)//4],
                         self.maturities[len(self.maturities)//2],
                         self.maturities[3*len(self.maturities)//4]]
        
        for T in maturities:
            strikes, ivs = self.get_skew(T)
            if self.S is not None:
                moneyness = strikes / self.S
                ax.plot(moneyness, ivs * 100, label=f'T={T*365:.0f}d')
            else:
                ax.plot(strikes, ivs * 100, label=f'T={T*365:.0f}d')
        
        if self.S is not None:
            ax.axvline(1.0, color='gray', linestyle='--', label='ATM')
            ax.set_xlabel('Moneyness (K/S)')
        else:
            ax.set_xlabel('Strike Price')
        
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title('Volatility Skew')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert surface to DataFrame format.
        
        Returns:
        --------
        DataFrame : IV surface data
        """
        data = []
        for i, K in enumerate(self.strikes):
            for j, T in enumerate(self.maturities):
                if not np.isnan(self.iv_data[i, j]):
                    data.append({
                        'strike': K,
                        'maturity': T,
                        'maturity_days': T * 365,
                        'implied_vol': self.iv_data[i, j]
                    })
        
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'IVSurface':
        """
        Create IVSurface from DataFrame.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with 'strike', 'maturity', 'implied_vol' columns
        
        Returns:
        --------
        IVSurface : Implied volatility surface
        """
        strikes = df['strike'].unique()
        maturities = df['maturity'].unique()
        strikes.sort()
        maturities.sort()
        
        iv_data = np.zeros((len(strikes), len(maturities)))
        
        for _, row in df.iterrows():
            k_idx = np.where(strikes == row['strike'])[0][0]
            t_idx = np.where(maturities == row['maturity'])[0][0]
            iv_data[k_idx, t_idx] = row['implied_vol']
        
        return cls(strikes, maturities, iv_data)