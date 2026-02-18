"""Monte Carlo option pricing with variance reduction techniques."""

import numpy as np
from typing import Literal, Optional, Tuple
from scipy.stats import norm


class MonteCarloOption:
    """
    Monte Carlo option pricing with various variance reduction techniques.
    
    Supports European, Asian, and Lookback options.
    
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
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def european_call(self, n_sims: int = 100000, 
                      n_steps: int = 252,
                      antithetic: bool = True,
                      control_variate: bool = True) -> Tuple[float, float]:
        """
        Price European call option using Monte Carlo.
        
        Parameters:
        -----------
        n_sims : int
            Number of simulations
        n_steps : int
            Number of time steps
        antithetic : bool
            Use antithetic variates
        control_variate : bool
            Use control variate (delta-based)
        
        Returns:
        --------
        tuple : (price, standard_error)
        """
        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        if antithetic:
            n_sims = n_sims // 2
            z = np.random.standard_normal((n_sims, n_steps))
            z = np.vstack([z, -z])
        else:
            z = np.random.standard_normal((n_sims, n_steps))
        
        # Generate paths
        log_returns = drift + diffusion * z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = self.S * np.exp(log_prices)
        
        # Terminal prices
        S_T = prices[:, -1]
        
        # Payoffs
        payoffs = np.maximum(S_T - self.K, 0)
        
        # Control variate
        if control_variate:
            # Use geometric average as control
            geo_mean = self.S * np.exp(np.mean(log_prices, axis=1))
            control_payoffs = np.maximum(geo_mean - self.K, 0)
            
            # Analytical price for geometric Asian
            geo_price = self._geometric_asian_call_analytical()
            
            # Regression coefficient
            cov = np.cov(payoffs, control_payoffs)
            beta = cov[0, 1] / cov[1, 1]
            
            payoffs = payoffs - beta * (control_payoffs - geo_price)
        
        # Discount and calculate
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
        
        return price, std_err
    
    def european_put(self, n_sims: int = 100000,
                     n_steps: int = 252,
                     antithetic: bool = True) -> Tuple[float, float]:
        """
        Price European put option using Monte Carlo.
        
        Uses put-call parity for efficiency.
        
        Parameters:
        -----------
        n_sims : int
            Number of simulations
        n_steps : int
            Number of time steps
        antithetic : bool
            Use antithetic variates
        
        Returns:
        --------
        tuple : (price, standard_error)
        """
        call_price, std_err = self.european_call(n_sims, n_steps, antithetic)
        
        # Put-call parity: P = C - S + K*exp(-rT)
        put_price = call_price - self.S + self.K * np.exp(-self.r * self.T)
        
        return put_price, std_err
    
    def asian_call(self, n_sims: int = 100000,
                   n_steps: int = 252,
                   average_type: Literal['arithmetic', 'geometric'] = 'arithmetic',
                   antithetic: bool = True) -> Tuple[float, float]:
        """
        Price Asian call option using Monte Carlo.
        
        Parameters:
        -----------
        n_sims : int
            Number of simulations
        n_steps : int
            Number of time steps
        average_type : str
            'arithmetic' or 'geometric' average
        antithetic : bool
            Use antithetic variates
        
        Returns:
        --------
        tuple : (price, standard_error)
        """
        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        if antithetic:
            n_sims = n_sims // 2
            z = np.random.standard_normal((n_sims, n_steps))
            z = np.vstack([z, -z])
        else:
            z = np.random.standard_normal((n_sims, n_steps))
        
        # Generate paths
        log_returns = drift + diffusion * z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = self.S * np.exp(log_prices)
        
        # Calculate average
        if average_type == 'arithmetic':
            avg_prices = np.mean(prices, axis=1)
        else:
            avg_prices = np.exp(np.mean(log_prices, axis=1))
        
        # Payoffs
        payoffs = np.maximum(avg_prices - self.K, 0)
        
        # Discount
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
        
        return price, std_err
    
    def asian_put(self, n_sims: int = 100000,
                  n_steps: int = 252,
                  average_type: Literal['arithmetic', 'geometric'] = 'arithmetic') -> Tuple[float, float]:
        """Price Asian put option."""
        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        z = np.random.standard_normal((n_sims, n_steps))
        log_returns = drift + diffusion * z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = self.S * np.exp(log_prices)
        
        if average_type == 'arithmetic':
            avg_prices = np.mean(prices, axis=1)
        else:
            avg_prices = np.exp(np.mean(log_prices, axis=1))
        
        payoffs = np.maximum(self.K - avg_prices, 0)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        return np.mean(discounted_payoffs), np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sims)
    
    def lookback_call(self, n_sims: int = 100000,
                      n_steps: int = 252,
                      lookback_type: Literal['fixed', 'floating'] = 'fixed') -> Tuple[float, float]:
        """
        Price Lookback call option using Monte Carlo.
        
        Parameters:
        -----------
        n_sims : int
            Number of simulations
        n_steps : int
            Number of time steps
        lookback_type : str
            'fixed' (strike is min price) or 'floating' (payoff is max - S_T)
        
        Returns:
        --------
        tuple : (price, standard_error)
        """
        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        z = np.random.standard_normal((n_sims, n_steps))
        log_returns = drift + diffusion * z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = self.S * np.exp(log_prices)
        
        if lookback_type == 'fixed':
            # Payoff: max(S_T - min(S), 0)
            min_prices = np.min(prices, axis=1)
            S_T = prices[:, -1]
            payoffs = np.maximum(S_T - min_prices, 0)
        else:
            # Payoff: max(S) - S_T
            max_prices = np.max(prices, axis=1)
            S_T = prices[:, -1]
            payoffs = max_prices - S_T
        
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sims)
        
        return price, std_err
    
    def lookback_put(self, n_sims: int = 100000,
                     n_steps: int = 252,
                     lookback_type: Literal['fixed', 'floating'] = 'fixed') -> Tuple[float, float]:
        """Price Lookback put option."""
        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        z = np.random.standard_normal((n_sims, n_steps))
        log_returns = drift + diffusion * z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = self.S * np.exp(log_prices)
        
        if lookback_type == 'fixed':
            max_prices = np.max(prices, axis=1)
            S_T = prices[:, -1]
            payoffs = np.maximum(max_prices - S_T, 0)
        else:
            min_prices = np.min(prices, axis=1)
            S_T = prices[:, -1]
            payoffs = S_T - min_prices
        
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sims)
        
        return price, std_err
    
    def barrier_call(self, barrier: float,
                     barrier_type: Literal['up-and-out', 'up-and-in', 
                                          'down-and-out', 'down-and-in'],
                     n_sims: int = 100000,
                     n_steps: int = 252) -> Tuple[float, float]:
        """
        Price Barrier call option using Monte Carlo.
        
        Parameters:
        -----------
        barrier : float
            Barrier level
        barrier_type : str
            Type of barrier option
        n_sims : int
            Number of simulations
        n_steps : int
            Number of time steps
        
        Returns:
        --------
        tuple : (price, standard_error)
        """
        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        z = np.random.standard_normal((n_sims, n_steps))
        log_returns = drift + diffusion * z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = self.S * np.exp(log_prices)
        
        S_T = prices[:, -1]
        
        # Check barrier conditions
        if barrier_type == 'up-and-out':
            # Knocked out if price exceeds barrier
            knocked_out = np.any(prices >= barrier, axis=1)
            payoffs = np.where(knocked_out, 0, np.maximum(S_T - self.K, 0))
        elif barrier_type == 'up-and-in':
            # Only pays if price exceeds barrier
            knocked_in = np.any(prices >= barrier, axis=1)
            payoffs = np.where(knocked_in, np.maximum(S_T - self.K, 0), 0)
        elif barrier_type == 'down-and-out':
            knocked_out = np.any(prices <= barrier, axis=1)
            payoffs = np.where(knocked_out, 0, np.maximum(S_T - self.K, 0))
        elif barrier_type == 'down-and-in':
            knocked_in = np.any(prices <= barrier, axis=1)
            payoffs = np.where(knocked_in, np.maximum(S_T - self.K, 0), 0)
        else:
            raise ValueError(f"Invalid barrier type: {barrier_type}")
        
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sims)
        
        return price, std_err
    
    def get_paths(self, n_sims: int = 1000,
                  n_steps: int = 252) -> np.ndarray:
        """
        Generate and return price paths for visualization.
        
        Parameters:
        -----------
        n_sims : int
            Number of paths
        n_steps : int
            Number of time steps
        
        Returns:
        --------
        ndarray : Price paths (n_sims x n_steps)
        """
        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        z = np.random.standard_normal((n_sims, n_steps))
        log_returns = drift + diffusion * z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = self.S * np.exp(log_prices)
        
        # Add initial price
        prices = np.column_stack([np.full(n_sims, self.S), prices])
        
        return prices
    
    def _geometric_asian_call_analytical(self) -> float:
        """
        Analytical price for geometric Asian call (for control variate).
        
        Returns:
        --------
        float : Analytical price
        """
        # Adjusted parameters for geometric average
        sigma_adj = self.sigma / np.sqrt(3)
        b = 0.5 * (self.r - 0.5 * self.sigma**2 + sigma_adj**2)
        
        d1 = (np.log(self.S / self.K) + (b + 0.5 * sigma_adj**2) * self.T) / \
             (sigma_adj * np.sqrt(self.T))
        d2 = d1 - sigma_adj * np.sqrt(self.T)
        
        return np.exp(-self.r * self.T) * (
            self.S * np.exp(b * self.T) * norm.cdf(d1) - 
            self.K * norm.cdf(d2)
        )