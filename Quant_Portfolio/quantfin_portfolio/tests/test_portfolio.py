"""Unit tests for portfolio module."""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from quantfin.portfolio.optimization import PortfolioOptimizer
from quantfin.portfolio.risk import RiskMetrics, RiskParity
from quantfin.portfolio.backtest import PortfolioBacktest, StrategyBacktest


@pytest.fixture
def sample_returns():
    """Generate sample returns data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
    n = len(dates)
    
    # Generate correlated returns
    data = {
        'AAPL': np.random.normal(0.0005, 0.02, n),
        'MSFT': np.random.normal(0.0004, 0.018, n),
        'GOOGL': np.random.normal(0.0006, 0.022, n),
        'AMZN': np.random.normal(0.0003, 0.025, n),
        'META': np.random.normal(0.0004, 0.028, n)
    }
    
    return pd.DataFrame(data, index=dates)


class TestPortfolioOptimizer:
    """Test portfolio optimization."""
    
    def test_max_sharpe(self, sample_returns):
        """Test maximum Sharpe optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        weights, info = optimizer.optimize_max_sharpe()
        
        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6)
        
        # All weights should be non-negative
        assert np.all(weights >= 0)
        
        # Should have positive expected return
        assert info['return'] > 0
        
        # Should have positive Sharpe
        assert info['sharpe'] > 0
    
    def test_min_variance(self, sample_returns):
        """Test minimum variance optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        weights, info = optimizer.optimize_min_variance()
        
        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6)
        
        # Should have lower variance than equal weight
        equal_var = sample_returns.var().mean() * 252
        assert info['risk'] < np.sqrt(equal_var)
    
    def test_target_return(self, sample_returns):
        """Test target return optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        target = 0.12
        weights, info = optimizer.optimize_mean_variance(target_return=target)
        
        # Should achieve target return
        assert np.isclose(info['return'], target, rtol=1e-2)
    
    def test_risk_parity(self, sample_returns):
        """Test risk parity optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        weights, info = optimizer.optimize_risk_parity()
        
        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6)
        
        # Risk contributions should be roughly equal
        rc = info['risk_contributions']
        rc_std = rc.std()
        rc_mean = rc.mean()
        assert rc_std / rc_mean < 0.5  # Less than 50% variation
    
    def test_cvar_optimization(self, sample_returns):
        """Test CVaR optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        weights, info = optimizer.optimize_cvar(confidence=0.95)
        
        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6)
        
        # Should have CVaR value
        assert 'cvar' in info
        assert info['cvar'] < 0  # CVaR should be negative (loss)
    
    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier generation."""
        optimizer = PortfolioOptimizer(sample_returns)
        frontier = optimizer.efficient_frontier(n_points=20)
        
        # Should have correct number of points
        assert len(frontier) == 20
        
        # Risk should increase with return
        assert frontier['risk'].is_monotonic_increasing or \
               (frontier['risk'].diff().dropna() >= -0.01).all()


class TestRiskMetrics:
    """Test risk metrics calculations."""
    
    def test_var_historical(self, sample_returns):
        """Test historical VaR."""
        risk = RiskMetrics(sample_returns)
        
        # Calculate VaR for single asset
        var_95 = risk.var_historical(confidence=0.95, weights=np.array([1, 0, 0, 0, 0]))
        
        # VaR should be negative (loss)
        assert var_95 < 0
        
        # 99% VaR should be more extreme than 95%
        var_99 = risk.var_historical(confidence=0.99, weights=np.array([1, 0, 0, 0, 0]))
        assert var_99 < var_95
    
    def test_var_parametric(self, sample_returns):
        """Test parametric VaR."""
        risk = RiskMetrics(sample_returns)
        
        var_95 = risk.var_parametric(confidence=0.95, weights=np.array([1, 0, 0, 0, 0]))
        
        # Should be negative
        assert var_95 < 0
    
    def test_cvar(self, sample_returns):
        """Test CVaR calculation."""
        risk = RiskMetrics(sample_returns)
        
        cvar_95 = risk.cvar(confidence=0.95, weights=np.array([1, 0, 0, 0, 0]))
        var_95 = risk.var_historical(confidence=0.95, weights=np.array([1, 0, 0, 0, 0]))
        
        # CVaR should be more extreme than VaR
        assert cvar_95 < var_95
    
    def test_maximum_drawdown(self, sample_returns):
        """Test maximum drawdown."""
        risk = RiskMetrics(sample_returns)
        
        max_dd = risk.maximum_drawdown(weights=np.array([1, 0, 0, 0, 0]))
        
        # Should be negative or zero
        assert max_dd <= 0
        
        # Should be reasonable (not -100%)
        assert max_dd > -1
    
    def test_beta(self, sample_returns):
        """Test beta calculation."""
        risk = RiskMetrics(sample_returns)
        
        # Use first asset as benchmark
        benchmark = sample_returns.iloc[:, 0]
        
        # Beta of asset with itself should be 1
        beta = risk.beta(benchmark, weights=np.array([1, 0, 0, 0, 0]))
        assert np.isclose(beta, 1.0, rtol=1e-2)
    
    def test_summary(self, sample_returns):
        """Test risk summary."""
        risk = RiskMetrics(sample_returns)
        
        summary = risk.summary(weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        
        # Should have expected columns
        assert 'Volatility (Ann %)' in summary.index
        assert 'VaR 95% (%)' in summary.index
        assert 'Max Drawdown (%)' in summary.index


class TestRiskParity:
    """Test risk parity calculations."""
    
    def test_risk_parity_weights(self, sample_returns):
        """Test risk parity weight calculation."""
        rp = RiskParity(sample_returns)
        weights = rp.get_weights()
        
        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6)
        
        # All weights should be positive
        assert np.all(weights > 0)
    
    def test_equal_risk_contribution(self, sample_returns):
        """Test that risk contributions are equal."""
        rp = RiskParity(sample_returns)
        weights = rp.get_weights()
        
        rc_df = rp.get_risk_contributions(weights)
        
        # All risk contributions should be roughly equal
        rc_values = rc_df['Risk Contribution (%)'].values
        assert np.std(rc_values) / np.mean(rc_values) < 0.3


class TestPortfolioBacktest:
    """Test portfolio backtesting."""
    
    def test_equal_weight_backtest(self, sample_returns):
        """Test equal weight backtest."""
        backtest = StrategyBacktest(sample_returns, rebalance_freq='M', transaction_cost=0.001)
        
        result = backtest.equal_weight(initial_capital=100000)
        
        # Should have results
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        
        # Final equity should be positive
        assert result.equity_curve.iloc[-1] > 0
        
        # Should have metrics
        assert 'total_return' in result.metrics
        assert 'sharpe_ratio' in result.metrics
    
    def test_inverse_volatility(self, sample_returns):
        """Test inverse volatility backtest."""
        backtest = StrategyBacktest(sample_returns, rebalance_freq='M', transaction_cost=0.001)
        
        result = backtest.inverse_volatility(window=63)
        
        # Should complete successfully
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
    
    def test_risk_parity_backtest(self, sample_returns):
        """Test risk parity backtest."""
        backtest = StrategyBacktest(sample_returns, rebalance_freq='M', transaction_cost=0.001)
        
        result = backtest.risk_parity(window=63)
        
        # Should complete successfully
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
    
    def test_transaction_costs(self, sample_returns):
        """Test that transaction costs reduce returns."""
        backtest_low = StrategyBacktest(sample_returns, rebalance_freq='M', transaction_cost=0.0001)
        backtest_high = StrategyBacktest(sample_returns, rebalance_freq='M', transaction_cost=0.01)
        
        result_low = backtest_low.equal_weight()
        result_high = backtest_high.equal_weight()
        
        # Higher costs should reduce returns
        assert result_high.metrics['total_return'] <= result_low.metrics['total_return']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])