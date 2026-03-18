"""Configuration management for QuantFin."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class Config:
    """Global configuration for QuantFin library."""
    
    # Data settings
    data_dir: str = field(default="./data")
    cache_dir: str = field(default="./cache")
    use_cache: bool = field(default=True)
    
    # API Keys (load from environment)
    polygon_api_key: Optional[str] = field(default=None)
    alpha_vantage_key: Optional[str] = field(default=None)
    
    # Trading settings
    default_commission: float = field(default=0.001)  # 10 bps
    default_slippage: float = field(default=0.0005)   # 5 bps
    risk_free_rate: float = field(default=0.045)      # 4.5%
    
    # Optimization settings
    optimization_tolerance: float = field(default=1e-9)
    max_optimization_iters: int = field(default=1000)
    
    # Monte Carlo settings
    default_mc_sims: int = field(default=100000)
    default_mc_steps: int = field(default=252)
    
    # Backtesting settings
    default_initial_capital: float = field(default=100000.0)
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.polygon_api_key is None:
            self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        if self.alpha_vantage_key is None:
            self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def get_cache_path(self, filename: str) -> str:
        """Get full path for cache file."""
        return os.path.join(self.cache_dir, filename)
    
    def get_data_path(self, filename: str) -> str:
        """Get full path for data file."""
        return os.path.join(self.data_dir, filename)