"""Black-Scholes option pricing model."""

import numpy as np
from scipy.stats import norm
from typing import Union, Literal


class BlackScholes:
    """
    Black-Scholes-Merton option pricing model.
    
    Prices European options on non-dividend paying stocks.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of underlying (annual)
    q : float, optional
        Dividend yield (annual), default 0
    """
    
    def __init__(self, S: float, K: float, T: float, 
                 r: float, sigma: float, q: float = 0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        
        # Precompute d1 and d2
        self._d1 = None
        self._d2 = None
    
    def _calculate_d1(self) -> float:
        """Calculate d1 parameter."""
        if self._d1 is None:
            self._d1 = (np.log(self.S / self.K) + 
                       (self.r - self.q + 0.5 * self.sigma**2) * self.T) / \
                      (self.sigma * np.sqrt(self.T))
        return self._d1
    
    def _calculate_d2(self) -> float:
        """Calculate d2 parameter."""
        if self._d2 is None:
            self._d2 = self._calculate_d1() - self.sigma * np.sqrt(self.T)
        return self._d2
    
    def call_price(self) -> float:
        """
        Calculate European call option price.
        
        Returns:
        --------
        float : Call option price
        """
        if self.T <= 0:
            return max(self.S - self.K, 0)
        
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()
        
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
    
    def put_price(self) -> float:
        """
        Calculate European put option price.
        
        Returns:
        --------
        float : Put option price
        """
        if self.T <= 0:
            return max(self.K - self.S, 0)
        
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()
        
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - 
                self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
    
    def price(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate option price.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Option price
        """
        if option_type.lower() == 'call':
            return self.call_price()
        else:
            return self.put_price()
    
    def delta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate delta (hedge ratio).
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Delta value
        """
        d1 = self._calculate_d1()
        
        if option_type.lower() == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            return np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
    
    def gamma(self) -> float:
        """
        Calculate gamma (rate of change of delta).
        
        Returns:
        --------
        float : Gamma value
        """
        d1 = self._calculate_d1()
        return (np.exp(-self.q * self.T) * norm.pdf(d1) / 
                (self.S * self.sigma * np.sqrt(self.T)))
    
    def vega(self) -> float:
        """
        Calculate vega (sensitivity to volatility).
        
        Returns:
        --------
        float : Vega value (for 1% change in vol)
        """
        d1 = self._calculate_d1()
        return (self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * 
                np.sqrt(self.T) * 0.01)
    
    def theta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate theta (time decay).
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Theta value (daily)
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()
        
        common_term = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * 
                        self.sigma) / (2 * np.sqrt(self.T))
        
        if option_type.lower() == 'call':
            theta = (common_term - 
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2) +
                    self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1))
        else:
            theta = (common_term + 
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) -
                    self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
        
        return theta / 365  # Convert to daily
    
    def rho(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate rho (sensitivity to interest rates).
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Rho value (for 1% change in rate)
        """
        d2 = self._calculate_d2()
        
        if option_type.lower() == 'call':
            return (self.K * self.T * np.exp(-self.r * self.T) * 
                   norm.cdf(d2) * 0.01)
        else:
            return (-self.K * self.T * np.exp(-self.r * self.T) * 
                   norm.cdf(-d2) * 0.01)
    
    def vanna(self) -> float:
        """
        Calculate vanna (d delta / d sigma).
        
        Returns:
        --------
        float : Vanna value
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()
        return -(d2 / self.sigma) * norm.pdf(d1) * np.exp(-self.q * self.T)
    
    def charm(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate charm (d delta / d T).
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Charm value
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()
        
        charm = (self.q * np.exp(-self.q * self.T) * norm.cdf(d1) -
                np.exp(-self.q * self.T) * norm.pdf(d1) * 
                (2 * (self.r - self.q) * self.T - d2 * self.sigma * np.sqrt(self.T)) /
                (2 * self.T * self.sigma * np.sqrt(self.T)))
        
        if option_type.lower() == 'put':
            charm -= self.q * np.exp(-self.q * self.T)
        
        return charm / 365  # Daily
    
    def all_greeks(self, option_type: Literal['call', 'put'] = 'call') -> dict:
        """
        Calculate all Greeks at once.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        dict : Dictionary of all Greeks
        """
        return {
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type),
            'vanna': self.vanna(),
            'charm': self.charm(option_type)
        }
    
    @staticmethod
    def from_market_price(S: float, K: float, T: float, r: float,
                          market_price: float, 
                          option_type: Literal['call', 'put'] = 'call',
                          q: float = 0.0) -> 'BlackScholes':
        """
        Create BlackScholes instance from market price (calibrates sigma).
        
        Parameters:
        -----------
        S, K, T, r : float
            Standard BS parameters
        market_price : float
            Observed market price
        option_type : str
            'call' or 'put'
        q : float
            Dividend yield
        
        Returns:
        --------
        BlackScholes : Calibrated model
        """
        from scipy.optimize import brentq
        
        def objective(sigma):
            bs = BlackScholes(S, K, T, r, sigma, q)
            return bs.price(option_type) - market_price
        
        try:
            sigma = brentq(objective, 1e-6, 5.0)
            return BlackScholes(S, K, T, r, sigma, q)
        except ValueError:
            raise ValueError("Could not calibrate volatility to market price")