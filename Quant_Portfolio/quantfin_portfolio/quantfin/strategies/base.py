"""Base classes for trading strategies."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    """Trading signal container."""
    ticker: str
    direction: int  # 1 for long, -1 for short, 0 for neutral
    strength: float  # Signal strength (0 to 1)
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str = "Strategy"):
        self.name = name
        self.signals: List[Signal] = []
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Fit the strategy to historical data.
        
        Parameters:
        -----------
        data : DataFrame
            Historical price/returns data
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals.
        
        Parameters:
        -----------
        data : DataFrame
            Current market data
        
        Returns:
        --------
        list : List of Signal objects
        """
        pass
    
    @abstractmethod
    def get_weights(self, data: pd.DataFrame) -> pd.Series:
        """
        Get portfolio weights.
        
        Parameters:
        -----------
        data : DataFrame
            Current market data
        
        Returns:
        --------
        Series : Portfolio weights by asset
        """
        pass
    
    def get_positions(self, capital: float, 
                     prices: pd.Series) -> pd.Series:
        """
        Calculate positions from weights.
        
        Parameters:
        -----------
        capital : float
            Available capital
        prices : Series
            Current prices
        
        Returns:
        --------
        Series : Number of shares/contracts
        """
        weights = self.get_weights(prices.to_frame().T)
        positions = (capital * weights) / prices
        return positions.fillna(0)


class TechnicalStrategy(Strategy):
    """Base class for technical indicator-based strategies."""
    
    def __init__(self, name: str = "TechnicalStrategy",
                 lookback: int = 20):
        super().__init__(name)
        self.lookback = lookback


class StatisticalStrategy(Strategy):
    """Base class for statistical arbitrage strategies."""
    
    def __init__(self, name: str = "StatisticalStrategy",
                 window: int = 63,
                 z_threshold: float = 2.0):
        super().__init__(name)
        self.window = window
        self.z_threshold = z_threshold


class MLStrategy(Strategy):
    """Base class for machine learning strategies."""
    
    def __init__(self, name: str = "MLStrategy",
                 model=None,
                 features: List[str] = None):
        super().__init__(name)
        self.model = model
        self.features = features or []
        self.predictions = []
    
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        pass
    
    def fit(self, data: pd.DataFrame):
        """Fit ML model."""
        X = self.prepare_features(data)
        # Subclasses should implement target preparation
        pass