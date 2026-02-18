"""Simulated broker for backtesting."""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class BrokerConfig:
    """Broker configuration."""
    commission_rate: float = 0.001  # 10 bps
    min_commission: float = 1.0
    slippage_model: str = 'fixed'  # 'fixed', 'percentage', 'volatility'
    slippage_value: float = 0.0005  # 5 bps
    margin_requirement: float = 0.5
    interest_rate: float = 0.02


class SimulatedBroker:
    """
    Simulated broker for realistic backtesting.
    
    Parameters:
    -----------
    config : BrokerConfig
        Broker configuration
    """
    
    def __init__(self, config: Optional[BrokerConfig] = None):
        self.config = config or BrokerConfig()
    
    def calculate_commission(self, quantity: float, 
                            price: float) -> float:
        """
        Calculate commission for trade.
        
        Parameters:
        -----------
        quantity : float
            Number of shares/contracts
        price : float
            Trade price
        
        Returns:
        --------
        float : Commission amount
        """
        trade_value = abs(quantity) * price
        commission = trade_value * self.config.commission_rate
        return max(commission, self.config.min_commission)
    
    def calculate_slippage(self, quantity: float,
                          price: float,
                          volatility: Optional[float] = None) -> float:
        """
        Calculate slippage for trade.
        
        Parameters:
        -----------
        quantity : float
            Number of shares/contracts
        price : float
            Trade price
        volatility : float, optional
            Asset volatility for volatility-based slippage
        
        Returns:
        --------
        float : Slippage amount (price adjustment)
        """
        if self.config.slippage_model == 'fixed':
            return self.config.slippage_value * price
        elif self.config.slippage_model == 'percentage':
            return self.config.slippage_value * price
        elif self.config.slippage_model == 'volatility' and volatility:
            return volatility * price * 0.1  # 10% of daily vol
        else:
            return self.config.slippage_value * price
    
    def get_fill_price(self, intended_price: float,
                      quantity: float,
                      side: str,
                      volatility: Optional[float] = None) -> float:
        """
        Get actual fill price including slippage.
        
        Parameters:
        -----------
        intended_price : float
            Target price
        quantity : float
            Trade size
        side : str
            'buy' or 'sell'
        volatility : float, optional
            Asset volatility
        
        Returns:
        --------
        float : Fill price
        """
        slippage = self.calculate_slippage(quantity, intended_price, volatility)
        
        if side == 'buy':
            return intended_price + slippage
        else:
            return intended_price - slippage
    
    def check_margin(self, position_value: float,
                    account_equity: float) -> bool:
        """
        Check if position meets margin requirements.
        
        Parameters:
        -----------
        position_value : float
            Total position value
        account_equity : float
            Account equity
        
        Returns:
        --------
        bool : True if margin requirement met
        """
        required = position_value * self.config.margin_requirement
        return account_equity >= required
    
    def calculate_buying_power(self, cash: float,
                              position_value: float) -> float:
        """
        Calculate available buying power.
        
        Parameters:
        -----------
        cash : float
            Available cash
        position_value : float
            Current position value
        
        Returns:
        --------
        float : Buying power
        """
        # Simplified: cash + margin allowance
        margin_allowance = position_value * (1 - self.config.margin_requirement)
        return cash + margin_allowance