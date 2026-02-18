"""Unit tests for core module."""

import pytest
import numpy as np
import pandas as pd
import os

import sys
sys.path.append('..')
from quantfin.core.config import Config
from quantfin.core.utils import (
    sharpe_ratio, maximum_drawdown, calmar_ratio,
    var_historical, cvar_historical, beta, alpha
)
from quantfin.core.data import SyntheticDataProvider, YahooFinanceProvider


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        assert config.risk_free_rate == 0.045
        assert config.default_commission == 0.001
        assert config.default_slippage == 0.0005
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = Config(
            risk_free_rate=0.03,
            default_commission=0.002
        )
        
        assert config.risk_free_rate == 0.03
        assert config.default_commission == 0.002
    
    def test_json_save_load(self, tmp_path):
        """Test saving and loading config from JSON."""
        config = Config(risk_free_rate=0.06)
        
        filepath = tmp_path / "config.json"
        config.to_json(str(filepath))
        
        loaded = Config.from_json(str(filepath))
        assert loaded.risk_free_rate == 0.06


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return np.random.normal(0.0005, 0.02, 252)
    
    @pytest.fixture
    def equity_curve(self, sample_returns):
        """Generate equity curve."""
        return np.cumprod(1 + sample_returns)
    
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = sharpe_ratio(sample_returns, risk_free_rate=0.045)
        
        # Should be a finite number
        assert np.isfinite(sharpe)
        
        # Higher returns should give higher Sharpe
        high_returns = sample_returns + 0.001
        high_sharpe = sharpe_ratio(high_returns, risk_free_rate=0.045)
        assert high_sharpe > sharpe
    
    def test_maximum_drawdown(self, equity_curve):
        """Test maximum drawdown calculation."""
        max_dd = maximum_drawdown(equity_curve)
        
        # Should be negative or zero
        assert max_dd <= 0
        
        # Should be greater than -100%
        assert max_dd > -1
    
    def test_calmar_ratio(self, sample_returns, equity_curve):
        """Test Calmar ratio calculation."""
        calmar = calmar_ratio(sample_returns)
        
        # Should be finite
        assert np.isfinite(calmar)
    
    def test_var_historical(self, sample_returns):
        """Test historical VaR."""
        var_95 = var_historical(sample_returns, confidence=0.95)
        
        # Should be negative (loss)
        assert var_95 < 0
        
        # 99% VaR should be more extreme
        var_99 = var_historical(sample_returns, confidence=0.99)
        assert var_99 < var_95
    
    def test_cvar_historical(self, sample_returns):
        """Test historical CVaR."""
        cvar_95 = cvar_historical(sample_returns, confidence=0.95)
        var_95 = var_historical(sample_returns, confidence=0.95)
        
        # CVaR should be more extreme than VaR
        assert cvar_95 < var_95
    
    def test_beta(self):
        """Test beta calculation."""
        np.random.seed(42)
        market = np.random.normal(0.0005, 0.015, 252)
        
        # Asset with beta = 1 (same as market)
        asset = market + np.random.normal(0, 0.005, 252)
        beta_val = beta(asset, market)
        
        assert 0.5 < beta_val < 1.5
        
        # Asset with higher volatility should have higher beta
        volatile_asset = 1.5 * market + np.random.normal(0, 0.01, 252)
        high_beta = beta(volatile_asset, market)
        assert high_beta > beta_val
    
    def test_alpha(self):
        """Test alpha calculation."""
        np.random.seed(42)
        market = np.random.normal(0.0005, 0.015, 252)
        
        # Asset with positive alpha
        asset = market + 0.0002 + np.random.normal(0, 0.005, 252)
        alpha_val = alpha(asset, market, risk_free_rate=0.045)
        
        # Should be positive
        assert alpha_val > 0


class TestDataProviders:
    """Test data providers."""
    
    def test_synthetic_data_provider(self):
        """Test synthetic data generation."""
        provider = SyntheticDataProvider()
        
        prices = provider.get_prices(
            tickers=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            seed=42
        )
        
        # Should have correct columns
        assert list(prices.columns) == ['AAPL', 'MSFT']
        
        # Should have positive prices
        assert (prices > 0).all().all()
        
        # Should have correct date range
        assert prices.index[0].year == 2020
    
    def test_synthetic_returns(self):
        """Test synthetic returns generation."""
        provider = SyntheticDataProvider()
        
        returns = provider.get_returns(
            tickers=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            seed=42
        )
        
        # Returns should have mean close to 0
        assert abs(returns.mean().mean()) < 0.01
    
    def test_data_manager_universe(self):
        """Test data manager universe selection."""
        from quantfin.core.data import DataManager
        
        provider = SyntheticDataProvider()
        manager = DataManager(provider)
        
        # Should be able to get universe data
        prices = manager.get_universe_data(
            universe='tech_giants',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Should have tech stocks
        assert len(prices.columns) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_sharpe_zero_volatility(self):
        """Test Sharpe with zero volatility."""
        returns = np.ones(252) * 0.001  # Constant returns
        sharpe = sharpe_ratio(returns)
        
        # Should handle gracefully
        assert sharpe == 0.0 or np.isfinite(sharpe)
    
    def test_var_single_observation(self):
        """Test VaR with single observation."""
        returns = np.array([0.01])
        var = var_historical(returns, confidence=0.95)
        
        # Should return the single value
        assert var == 0.01
    
    def test_drawdown_single_point(self):
        """Test drawdown with single point."""
        equity = np.array([1.0])
        max_dd = maximum_drawdown(equity)
        
        # Should be 0 (no drawdown)
        assert max_dd == 0.0
    
    def test_beta_identical_returns(self):
        """Test beta with identical returns."""
        returns = np.ones(252) * 0.001
        beta_val = beta(returns, returns)
        
        # Should handle gracefully
        assert np.isfinite(beta_val) or beta_val == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])