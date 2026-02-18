"""Trinomial option pricing model."""

import numpy as np
from typing import Literal


class TrinomialModel:
    """
    Trinomial option pricing model.
    
    More accurate than binomial for same number of steps.
    Supports European and American options.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    n : int
        Number of time steps
    """
    
    def __init__(self, S: float, K: float, T: float, 
                 r: float, sigma: float, n: int = 100):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n = n
        
        # Calculate tree parameters
        self.dt = T / n
        self.dx = sigma * np.sqrt(3 * self.dt)
        self.nu = r - 0.5 * sigma**2
        
        # Probabilities
        a = (sigma**2 * self.dt + self.nu**2 * self.dt**2) / self.dx**2
        b = self.nu * self.dt / self.dx
        
        self.pu = 0.5 * (a + b)
        self.pd = 0.5 * (a - b)
        self.pm = 1.0 - a
        
        self.discount = np.exp(-r * self.dt)
    
    def call_price(self, american: bool = False) -> float:
        """
        Calculate call option price.
        
        Parameters:
        -----------
        american : bool
            If True, price American option
        
        Returns:
        --------
        float : Call option price
        """
        return self._price_option('call', american)
    
    def put_price(self, american: bool = False) -> float:
        """
        Calculate put option price.
        
        Parameters:
        -----------
        american : bool
            If True, price American option
        
        Returns:
        --------
        float : Put option price
        """
        return self._price_option('put', american)
    
    def price(self, option_type: Literal['call', 'put'] = 'call',
              american: bool = False) -> float:
        """
        Calculate option price.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        american : bool
            If True, price American option
        
        Returns:
        --------
        float : Option price
        """
        return self._price_option(option_type, american)
    
    def _price_option(self, option_type: str, american: bool) -> float:
        """Internal pricing method."""
        num_nodes = 2 * self.n + 1
        center = self.n
        
        # Terminal stock prices
        stock = self.S * np.exp(np.arange(-self.n, self.n + 1) * self.dx)
        
        # Terminal option values
        if option_type == 'call':
            v = np.maximum(stock - self.K, 0)
        else:
            v = np.maximum(self.K - stock, 0)
        
        # Backward induction
        for step in range(self.n):
            v_new = np.zeros(num_nodes)
            
            # Interior nodes
            for k in range(1, num_nodes - 1):
                cont = self.pu * v[k + 1] + self.pm * v[k] + self.pd * v[k - 1]
                v_new[k] = self.discount * cont
            
            # Boundary nodes (simplified)
            v_new[0] = self.discount * (self.pu * v[1] + self.pm * v[0] + self.pd * v[0])
            v_new[-1] = self.discount * (self.pu * v[-1] + self.pm * v[-1] + self.pd * v[-2])
            
            # Early exercise for American options
            if american:
                current_stock = self.S * np.exp(
                    (np.arange(-self.n + step, self.n - step + 1)) * self.dx
                )
                if option_type == 'call':
                    intrinsic = np.maximum(current_stock - self.K, 0)
                else:
                    intrinsic = np.maximum(self.K - current_stock, 0)
                v_new = np.maximum(v_new, intrinsic)
            
            v = v_new
        
        return v[center]
    
    def delta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate delta from trinomial tree.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Delta
        """
        num_nodes = 2 * self.n + 1
        center = self.n
        
        # Terminal values
        stock = self.S * np.exp(np.arange(-self.n, self.n + 1) * self.dx)
        if option_type == 'call':
            v = np.maximum(stock - self.K, 0)
        else:
            v = np.maximum(self.K - stock, 0)
        
        # One step backward
        v_new = np.zeros(num_nodes)
        for k in range(1, num_nodes - 1):
            v_new[k] = self.discount * (self.pu * v[k + 1] + 
                                        self.pm * v[k] + 
                                        self.pd * v[k - 1])
        
        # Delta at center
        stock_up = self.S * np.exp(self.dx)
        stock_down = self.S * np.exp(-self.dx)
        
        delta = (v_new[center - 1] - v_new[center + 1]) / (stock_up - stock_down)
        
        return delta
    
    def convergence_analysis(self, option_type: Literal['call', 'put'] = 'call',
                            steps_range: list = None) -> dict:
        """
        Analyze convergence as number of steps increases.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        steps_range : list, optional
            List of step counts to test
        
        Returns:
        --------
        dict : Convergence results
        """
        if steps_range is None:
            steps_range = [10, 25, 50, 100, 200, 500]
        
        results = {}
        for n in steps_range:
            model = TrinomialModel(self.S, self.K, self.T, self.r, self.sigma, n)
            results[n] = model.price(option_type)
        
        return results