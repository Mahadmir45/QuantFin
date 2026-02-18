"""Options strategies and combinations."""

import numpy as np
from typing import List, Literal, Tuple, Dict
from dataclasses import dataclass
from .models.black_scholes import BlackScholes


@dataclass
class OptionLeg:
    """Single option leg in a strategy."""
    strike: float
    maturity: float
    position: Literal['long', 'short']
    option_type: Literal['call', 'put']
    quantity: int = 1


class OptionStrategy:
    """
    Multi-leg options strategy.
    
    Parameters:
    -----------
    S : float
        Current stock price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    legs : list
        List of OptionLeg objects
    """
    
    def __init__(self, S: float, r: float, sigma: float, legs: List[OptionLeg]):
        self.S = S
        self.r = r
        self.sigma = sigma
        self.legs = legs
    
    def price(self) -> float:
        """
        Calculate total strategy price.
        
        Returns:
        --------
        float : Strategy price (net debit/credit)
        """
        total = 0.0
        
        for leg in self.legs:
            bs = BlackScholes(self.S, leg.strike, leg.maturity, 
                             self.r, self.sigma)
            leg_price = bs.price(leg.option_type)
            
            if leg.position == 'long':
                total += leg.quantity * leg_price
            else:
                total -= leg.quantity * leg_price
        
        return total
    
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calculate payoff at expiration for given spot prices.
        
        Parameters:
        -----------
        spot_prices : array-like
            Array of spot prices at expiration
        
        Returns:
        --------
        array : Payoff values
        """
        payoffs = np.zeros_like(spot_prices)
        
        for leg in self.legs:
            if leg.option_type == 'call':
                intrinsic = np.maximum(spot_prices - leg.strike, 0)
            else:
                intrinsic = np.maximum(leg.strike - spot_prices, 0)
            
            if leg.position == 'long':
                payoffs += leg.quantity * intrinsic
            else:
                payoffs -= leg.quantity * intrinsic
        
        return payoffs
    
    def pnl(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calculate P&L for given spot prices.
        
        Parameters:
        -----------
        spot_prices : array-like
            Array of spot prices
        
        Returns:
        --------
        array : P&L values
        """
        entry_cost = self.price()
        return self.payoff(spot_prices) - entry_cost
    
    def breakevens(self, spot_range: Tuple[float, float] = (0.5, 1.5),
                   n_points: int = 1000) -> List[float]:
        """
        Find breakeven points.
        
        Parameters:
        -----------
        spot_range : tuple
            (min, max) as multiples of current spot
        n_points : int
            Number of points to evaluate
        
        Returns:
        --------
        list : Breakeven spot prices
        """
        spots = np.linspace(spot_range[0] * self.S, spot_range[1] * self.S, n_points)
        pnls = self.pnl(spots)
        
        # Find zero crossings
        breakevens = []
        for i in range(len(pnls) - 1):
            if pnls[i] * pnls[i + 1] < 0:  # Sign change
                # Linear interpolation
                breakeven = spots[i] - pnls[i] * (spots[i + 1] - spots[i]) / (pnls[i + 1] - pnls[i])
                breakevens.append(breakeven)
        
        return breakevens
    
    def max_profit(self, spot_range: Tuple[float, float] = (0.5, 1.5)) -> float:
        """Calculate maximum profit."""
        spots = np.linspace(spot_range[0] * self.S, spot_range[1] * self.S, 1000)
        pnls = self.pnl(spots)
        return np.max(pnls)
    
    def max_loss(self, spot_range: Tuple[float, float] = (0.5, 1.5)) -> float:
        """Calculate maximum loss."""
        spots = np.linspace(spot_range[0] * self.S, spot_range[1] * self.S, 1000)
        pnls = self.pnl(spots)
        return np.min(pnls)


class SpreadStrategy:
    """Factory for common spread strategies."""
    
    @staticmethod
    def bull_call_spread(S: float, K1: float, K2: float, T: float, 
                        r: float, sigma: float) -> OptionStrategy:
        """
        Create bull call spread: Buy call at K1, sell call at K2 (K1 < K2).
        
        Parameters:
        -----------
        S : float
            Current stock price
        K1 : float
            Lower strike (long)
        K2 : float
            Higher strike (short)
        T : float
            Time to expiration
        r : float
            Risk-free rate
        sigma : float
            Volatility
        
        Returns:
        --------
        OptionStrategy : Bull call spread
        """
        legs = [
            OptionLeg(K1, T, 'long', 'call'),
            OptionLeg(K2, T, 'short', 'call')
        ]
        return OptionStrategy(S, r, sigma, legs)
    
    @staticmethod
    def bear_put_spread(S: float, K1: float, K2: float, T: float,
                       r: float, sigma: float) -> OptionStrategy:
        """Create bear put spread: Buy put at K2, sell put at K1 (K1 < K2)."""
        legs = [
            OptionLeg(K2, T, 'long', 'put'),
            OptionLeg(K1, T, 'short', 'put')
        ]
        return OptionStrategy(S, r, sigma, legs)
    
    @staticmethod
    def iron_condor(S: float, K1: float, K2: float, K3: float, K4: float,
                   T: float, r: float, sigma: float) -> OptionStrategy:
        """
        Create iron condor: Sell put spread and call spread.
        
        K1 < K2 < K3 < K4
        Sell put at K2, buy put at K1
        Sell call at K3, buy call at K4
        """
        legs = [
            OptionLeg(K1, T, 'long', 'put'),
            OptionLeg(K2, T, 'short', 'put'),
            OptionLeg(K3, T, 'short', 'call'),
            OptionLeg(K4, T, 'long', 'call')
        ]
        return OptionStrategy(S, r, sigma, legs)
    
    @staticmethod
    def butterfly_call(S: float, K1: float, K2: float, K3: float,
                      T: float, r: float, sigma: float) -> OptionStrategy:
        """
        Create long call butterfly: Buy K1, sell 2x K2, buy K3.
        
        K1 < K2 < K3, typically K2 = (K1 + K3) / 2
        """
        legs = [
            OptionLeg(K1, T, 'long', 'call'),
            OptionLeg(K2, T, 'short', 'call', 2),
            OptionLeg(K3, T, 'long', 'call')
        ]
        return OptionStrategy(S, r, sigma, legs)
    
    @staticmethod
    def straddle(S: float, K: float, T: float, r: float, sigma: float) -> OptionStrategy:
        """Create long straddle: Buy call and put at same strike."""
        legs = [
            OptionLeg(K, T, 'long', 'call'),
            OptionLeg(K, T, 'long', 'put')
        ]
        return OptionStrategy(S, r, sigma, legs)
    
    @staticmethod
    def strangle(S: float, K1: float, K2: float, T: float, r: float, 
                sigma: float) -> OptionStrategy:
        """
        Create long strangle: Buy OTM put and OTM call.
        
        K1 < S < K2
        """
        legs = [
            OptionLeg(K1, T, 'long', 'put'),
            OptionLeg(K2, T, 'long', 'call')
        ]
        return OptionStrategy(S, r, sigma, legs)
    
    @staticmethod
    def calendar_spread(S: float, K: float, T1: float, T2: float,
                       r: float, sigma: float) -> OptionStrategy:
        """
        Create calendar spread: Sell near-term, buy longer-term.
        
        T1 < T2
        """
        legs = [
            OptionLeg(K, T1, 'short', 'call'),
            OptionLeg(K, T2, 'long', 'call')
        ]
        return OptionStrategy(S, r, sigma, legs)
    
    @staticmethod
    def collar(S: float, K_put: float, K_call: float, T: float,
              r: float, sigma: float) -> OptionStrategy:
        """
        Create collar: Own stock, buy protective put, sell covered call.
        
        K_put < S < K_call
        """
        legs = [
            OptionLeg(K_put, T, 'long', 'put'),
            OptionLeg(K_call, T, 'short', 'call')
        ]
        return OptionStrategy(S, r, sigma, legs)


class StrategyAnalyzer:
    """Analyze options strategies."""
    
    def __init__(self, strategy: OptionStrategy):
        self.strategy = strategy
    
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        max_profit = self.strategy.max_profit()
        max_loss = abs(self.strategy.max_loss())
        
        if max_loss == 0:
            return np.inf
        
        return max_profit / max_loss
    
    def probability_of_profit(self, mu: float = None, sigma: float = None,
                             n_sims: int = 100000) -> float:
        """
        Estimate probability of profit using Monte Carlo.
        
        Parameters:
        -----------
        mu : float, optional
            Expected return (default: risk-free rate)
        sigma : float, optional
            Volatility (default: strategy sigma)
        n_sims : int
            Number of simulations
        
        Returns:
        --------
        float : Probability of profit
        """
        if mu is None:
            mu = self.strategy.r
        if sigma is None:
            sigma = self.strategy.sigma
        
        # Get shortest maturity
        T = min(leg.maturity for leg in self.strategy.legs)
        
        # Simulate terminal prices
        z = np.random.standard_normal(n_sims)
        S_T = self.strategy.S * np.exp(
            (mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z
        )
        
        # Calculate P&L
        pnls = self.strategy.pnl(S_T)
        
        return np.mean(pnls > 0)
    
    def greeks(self) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for the strategy.
        
        Returns:
        --------
        dict : Dictionary of Greeks
        """
        greeks = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
        
        for leg in self.strategy.legs:
            from .greeks import GreeksCalculator
            bs = BlackScholes(self.strategy.S, leg.strike, leg.maturity,
                             self.strategy.r, self.strategy.sigma)
            calc = GreeksCalculator(bs)
            leg_greeks = calc.all_greeks(leg.option_type)
            
            multiplier = leg.quantity if leg.position == 'long' else -leg.quantity
            
            for greek in greeks:
                greeks[greek] += multiplier * leg_greeks[greek]
        
        return greeks