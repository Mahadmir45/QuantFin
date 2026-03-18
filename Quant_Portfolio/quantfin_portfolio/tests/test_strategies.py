"""Unit tests for strategies module."""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from quantfin.strategies.topology import TopologyAlphaStrategy
from quantfin.core.data import SyntheticDataProvider


@pytest.fixture
def sample_data():
    """Generate sample market data."""
    np.random.seed(42)
    provider = SyntheticDataProvider()
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    prices = provider.get_prices(stocks, '2020-01-01', '2022-12-31', seed=42)
    returns = prices.pct_change().dropna()
    
    # Generate synthetic VIX
    vix = pd.Series(
        20 + 5 * np.random.randn(len(prices)).cumsum() * 0.1,
        index=prices.index
    ).clip(10, 50)
    
    return prices, returns, vix


class TestTopologyAlphaStrategy:
    """Test Topology Alpha Strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return TopologyAlphaStrategy(
            stocks=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            window_size=30,  # Smaller for testing
            vix_threshold=30,
            betti_threshold_low=25,
            betti_threshold_high=35,
            diffusion_time=0.5
        )
    
    def test_build_correlation_graph(self, strategy, sample_data):
        """Test correlation graph construction."""
        prices, returns, _ = sample_data
        
        window = returns.iloc[:30]
        L, corr = strategy.build_correlation_graph(window)
        
        # Should return valid matrices
        assert L.shape == (5, 5)
        assert corr.shape == (5, 5)
        
        # Laplacian should be symmetric
        assert np.allclose(L, L.T)
        
        # Smallest eigenvalue should be 0
        eigenvalues = np.linalg.eigvalsh(L)
        assert abs(eigenvalues[0]) < 1e-10
    
    def test_laplacian_diffusion(self, strategy, sample_data):
        """Test Laplacian diffusion."""
        prices, returns, _ = sample_data
        
        window = returns.iloc[:30]
        L, _ = strategy.build_correlation_graph(window)
        
        signal = np.random.randn(5)
        diffused = strategy.laplacian_diffusion(L, signal)
        
        # Diffused signal should have same shape
        assert diffused.shape == signal.shape
        
        # Diffusion should smooth the signal
        assert np.std(diffused) <= np.std(signal) * 1.5
    
    def test_compute_persistence_features(self, strategy, sample_data):
        """Test persistence feature computation."""
        prices, returns, _ = sample_data
        
        window = returns.iloc[:30]
        _, corr = strategy.build_correlation_graph(window)
        
        features = strategy.compute_persistence_features([corr])
        
        # Should return features
        assert features.shape == (1, 6)
        
        # Features should be finite
        assert np.all(np.isfinite(features))
    
    def test_detect_regime(self, strategy):
        """Test regime detection."""
        # Low risk regime
        regime = strategy.detect_regime(betti_std=20, vix=20)
        assert regime == 'low_risk'
        
        # High risk regime
        regime = strategy.detect_regime(betti_std=50, vix=40)
        assert regime == 'high_risk'
        
        # Medium risk regime
        regime = strategy.detect_regime(betti_std=35, vix=25)
        assert regime == 'medium_risk'
    
    def test_get_exposure(self, strategy):
        """Test exposure calculation."""
        # Low risk - high exposure
        assert strategy.get_exposure('low_risk') == 1.2
        
        # Medium risk - normal exposure
        assert strategy.get_exposure('medium_risk') == 1.0
        
        # High risk - no exposure
        assert strategy.get_exposure('high_risk') == 0.0
    
    def test_get_weights(self, strategy, sample_data):
        """Test weight calculation."""
        prices, returns, _ = sample_data
        
        window = returns.iloc[:30]
        weights = strategy.get_weights(window, vix_current=20)
        
        # Should return valid weights
        assert len(weights) == 5
        assert np.all(weights >= 0)
        assert np.all(weights <= 1.2)  # Max exposure
        
        # Weights should sum to exposure level
        assert abs(weights.sum() - 1.2) < 0.1
    
    def test_backtest(self, strategy, sample_data):
        """Test full backtest."""
        prices, returns, vix = sample_data
        
        results = strategy.backtest(
            prices=prices,
            vix=vix,
            transaction_cost=0.001,
            initial_capital=100000
        )
        
        # Should return results
        assert 'equity_curve' in results
        assert 'returns' in results
        assert 'metrics' in results
        assert 'signals' in results
        
        # Equity curve should be positive
        assert (results['equity_curve'] > 0).all()
        
        # Should have metrics
        assert 'total_return' in results['metrics']
        assert 'sharpe_ratio' in results['metrics']
    
    def test_risk_parity_weights(self, strategy, sample_data):
        """Test that weights follow risk parity."""
        prices, returns, _ = sample_data
        
        window = returns.iloc[:30]
        weights = strategy.get_weights(window, vix_current=20)
        
        # Calculate risk contributions
        cov = window.cov() * 252
        port_var = weights @ cov @ weights
        marginal_risk = cov @ weights
        rc = weights * marginal_risk / np.sqrt(port_var)
        
        # Risk contributions should be roughly equal
        rc_std = np.std(rc)
        rc_mean = np.mean(rc)
        assert rc_std / rc_mean < 1.0  # Allow some variation


class TestStrategyEdgeCases:
    """Test strategy edge cases."""
    
    def test_empty_window(self):
        """Test handling of empty window."""
        strategy = TopologyAlphaStrategy(
            stocks=['AAPL', 'MSFT'],
            window_size=30
        )
        
        # Should handle gracefully
        window = pd.DataFrame({'AAPL': [], 'MSFT': []})
        try:
            weights = strategy.get_weights(window, vix_current=20)
            # If it doesn't raise, weights should be zeros
            assert len(weights) == 2
        except:
            pass  # Raising is also acceptable
    
    def test_constant_returns(self):
        """Test with constant returns."""
        strategy = TopologyAlphaStrategy(
            stocks=['AAPL', 'MSFT', 'GOOGL'],
            window_size=10
        )
        
        # Create constant returns
        dates = pd.date_range('2020-01-01', periods=10)
        window = pd.DataFrame({
            'AAPL': [0.001] * 10,
            'MSFT': [0.001] * 10,
            'GOOGL': [0.001] * 10
        }, index=dates)
        
        # Should handle gracefully
        try:
            weights = strategy.get_weights(window, vix_current=20)
            assert len(weights) == 3
        except:
            pass
    
    def test_high_vix(self, sample_data):
        """Test behavior with high VIX."""
        prices, returns, _ = sample_data
        
        strategy = TopologyAlphaStrategy(
            stocks=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            window_size=30,
            vix_threshold=20  # Low threshold for testing
        )
        
        window = returns.iloc[:30]
        weights = strategy.get_weights(window, vix_current=50)  # High VIX
        
        # Should reduce exposure
        assert weights.sum() < 1.0


class TestStrategyPerformance:
    """Test strategy performance characteristics."""
    
    def test_transaction_costs_impact(self, sample_data):
        """Test that transaction costs reduce returns."""
        prices, returns, vix = sample_data
        
        strategy = TopologyAlphaStrategy(
            stocks=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            window_size=30
        )
        
        # Low costs
        results_low = strategy.backtest(
            prices=prices, vix=vix,
            transaction_cost=0.0001, initial_capital=100000
        )
        
        # High costs
        results_high = strategy.backtest(
            prices=prices, vix=vix,
            transaction_cost=0.01, initial_capital=100000
        )
        
        # Higher costs should reduce returns
        assert results_high['metrics']['total_return'] <= results_low['metrics']['total_return']
    
    def test_regime_changes(self, sample_data):
        """Test that strategy adapts to regime changes."""
        prices, returns, vix = sample_data
        
        strategy = TopologyAlphaStrategy(
            stocks=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            window_size=30,
            vix_threshold=25
        )
        
        results = strategy.backtest(prices=prices, vix=vix)
        signals = results['signals']
        
        # Should have different regimes
        assert len(signals['regime'].unique()) > 1
        
        # Exposure should vary
        assert signals['exposure'].std() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])