"""Binomial option pricing model (Cox-Ross-Rubinstein)."""

import numpy as np
from typing import Literal


class BinomialModel:
    """
    Cox-Ross-Rubinstein binomial option pricing model.
    
    Supports both European and American options.
    
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
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-r * self.dt)
    
    def call_price(self, american: bool = False) -> float:
        """
        Calculate call option price.
        
        Parameters:
        -----------
        american : bool
            If True, price American option (allows early exercise)
        
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
        # Stock prices at maturity
        stock = self.S * (self.d ** np.arange(self.n, -1, -1)) * \
                (self.u ** np.arange(0, self.n + 1))
        
        # Option values at maturity
        if option_type == 'call':
            option = np.maximum(stock - self.K, 0)
        else:
            option = np.maximum(self.K - stock, 0)
        
        # Backward induction
        for i in range(self.n - 1, -1, -1):
            option = self.discount * (self.p * option[1:] + 
                                      (1 - self.p) * option[:-1])
            
            if american:
                stock = stock[:-1] / self.d
                if option_type == 'call':
                    intrinsic = np.maximum(stock - self.K, 0)
                else:
                    intrinsic = np.maximum(self.K - stock, 0)
                option = np.maximum(option, intrinsic)
        
        return option[0]
    
    def get_tree(self, option_type: Literal['call', 'put'] = 'call',
                 american: bool = False) -> tuple:
        """
        Get the full binomial tree.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        american : bool
            If True, build American option tree
        
        Returns:
        --------
        tuple : (stock_tree, option_tree)
        """
        # Initialize stock tree
        stock_tree = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n + 1):
            for j in range(i + 1):
                stock_tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)
        
        # Initialize option tree
        option_tree = np.zeros((self.n + 1, self.n + 1))
        if option_type == 'call':
            option_tree[:, self.n] = np.maximum(stock_tree[:, self.n] - self.K, 0)
        else:
            option_tree[:, self.n] = np.maximum(self.K - stock_tree[:, self.n], 0)
        
        # Backward induction
        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                continuation = self.discount * (self.p * option_tree[j, i + 1] + 
                                               (1 - self.p) * option_tree[j + 1, i + 1])
                
                if american:
                    if option_type == 'call':
                        intrinsic = max(stock_tree[j, i] - self.K, 0)
                    else:
                        intrinsic = max(self.K - stock_tree[j, i], 0)
                    option_tree[j, i] = max(continuation, intrinsic)
                else:
                    option_tree[j, i] = continuation
        
        return stock_tree, option_tree
    
    def delta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate delta from binomial tree.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Delta
        """
        stock_tree, option_tree = self.get_tree(option_type)
        
        # Delta at root
        delta = (option_tree[0, 1] - option_tree[1, 1]) / \
                (stock_tree[0, 1] - stock_tree[1, 1])
        
        return delta
    
    def gamma(self) -> float:
        """
        Calculate gamma from binomial tree.
        
        Returns:
        --------
        float : Gamma
        """
        stock_tree, option_tree = self.get_tree('call')
        
        # Deltas at time 1
        delta_up = (option_tree[0, 2] - option_tree[1, 2]) / \
                   (stock_tree[0, 2] - stock_tree[1, 2])
        delta_down = (option_tree[1, 2] - option_tree[2, 2]) / \
                     (stock_tree[1, 2] - stock_tree[2, 2])
        
        # Gamma
        gamma = (delta_up - delta_down) / \
                (0.5 * (stock_tree[0, 2] - stock_tree[2, 2]))
        
        return gamma
    
    def theta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate theta from binomial tree.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        float : Theta (daily)
        """
        stock_tree, option_tree = self.get_tree(option_type)
        
        # Theta approximation
        theta = (option_tree[1, 2] - option_tree[0, 0]) / (2 * self.dt)
        
        return theta / 365  # Convert to daily