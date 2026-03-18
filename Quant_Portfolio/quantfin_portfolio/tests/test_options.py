"""Unit tests for options pricing module."""

import pytest
import numpy as np
from scipy.stats import norm

import sys
sys.path.append('..')
from quantfin.options.models import BlackScholes, BinomialModel, TrinomialModel, MonteCarloOption
from quantfin.options.greeks import GreeksCalculator
from quantfin.options.implied_vol import ImpliedVolatility
from quantfin.options.strategies import SpreadStrategy, OptionStrategy, OptionLeg


class TestBlackScholes:
    """Test Black-Scholes model."""
    
    @pytest.fixture
    def bs_model(self):
        return BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    
    def test_call_price(self, bs_model):
        """Test call option pricing."""
        price = bs_model.call_price()
        assert price > 0
        assert price < 100  # Should be less than stock price
        
        # Compare with manual calculation
        d1 = (np.log(100/100) + (0.05 + 0.2**2/2) * 1) / (0.2 * np.sqrt(1))
        d2 = d1 - 0.2 * np.sqrt(1)
        expected = 100 * norm.cdf(d1) - 100 * np.exp(-0.05) * norm.cdf(d2)
        assert np.isclose(price, expected, rtol=1e-10)
    
    def test_put_price(self, bs_model):
        """Test put option pricing."""
        price = bs_model.put_price()
        assert price > 0
        
        # Put-call parity
        call = bs_model.call_price()
        parity = call - 100 + 100 * np.exp(-0.05)
        assert np.isclose(price, parity, rtol=1e-10)
    
    def test_greeks(self, bs_model):
        """Test Greeks calculation."""
        greeks = bs_model.all_greeks('call')
        
        # Delta should be between 0 and 1 for call
        assert 0 < greeks['delta'] < 1
        
        # Gamma should be positive
        assert greeks['gamma'] > 0
        
        # Vega should be positive
        assert greeks['vega'] > 0
        
        # Theta should be negative (time decay)
        assert greeks['theta'] < 0
        
        # Put delta should be negative
        put_greeks = bs_model.all_greeks('put')
        assert -1 < put_greeks['delta'] < 0
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Zero time to expiration
        bs = BlackScholes(S=100, K=100, T=0, r=0.05, sigma=0.2)
        assert bs.call_price() == max(100 - 100, 0)
        assert bs.put_price() == max(100 - 100, 0)
        
        # Deep ITM
        bs = BlackScholes(S=150, K=100, T=1.0, r=0.05, sigma=0.2)
        assert bs.call_price() > 40  # Should be close to intrinsic
        
        # Deep OTM
        bs = BlackScholes(S=50, K=100, T=1.0, r=0.05, sigma=0.2)
        assert bs.call_price() < 5  # Should be close to 0
    
    def test_from_market_price(self):
        """Test calibration from market price."""
        bs = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        market_price = bs.call_price()
        
        # Calibrate
        calibrated = BlackScholes.from_market_price(
            S=100, K=100, T=1.0, r=0.05,
            market_price=market_price, option_type='call'
        )
        
        assert np.isclose(calibrated.sigma, 0.2, rtol=1e-4)


class TestBinomialModel:
    """Test Binomial model."""
    
    def test_convergence_to_bs(self):
        """Test that binomial converges to Black-Scholes."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs = BlackScholes(S, K, T, r, sigma)
        bs_price = bs.call_price()
        
        # Test with increasing steps
        for n in [50, 100, 500, 1000]:
            binom = BinomialModel(S, K, T, r, sigma, n)
            bin_price = binom.call_price(american=False)
            error = abs(bin_price - bs_price)
            
            # Error should decrease with more steps
            assert error < 0.1
    
    def test_american_premium(self):
        """Test that American options have early exercise premium."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        binom = BinomialModel(S, K, T, r, sigma, 100)
        
        european_put = binom.put_price(american=False)
        american_put = binom.put_price(american=True)
        
        # American put should be worth more (early exercise valuable)
        assert american_put >= european_put
    
    def test_delta(self):
        """Test delta calculation from tree."""
        binom = BinomialModel(100, 100, 1.0, 0.05, 0.2, 100)
        delta = binom.delta('call')
        
        # Delta should be reasonable
        assert 0 < delta < 1


class TestMonteCarlo:
    """Test Monte Carlo option pricing."""
    
    def test_european_call(self):
        """Test European call pricing."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        mc = MonteCarloOption(S, K, T, r, sigma)
        price, err = mc.european_call(n_sims=50000)
        
        bs = BlackScholes(S, K, T, r, sigma)
        bs_price = bs.call_price()
        
        # Should be within 2 standard errors
        assert abs(price - bs_price) < 2 * err
    
    def test_put_call_parity_mc(self):
        """Test put-call parity with Monte Carlo."""
        mc = MonteCarloOption(100, 100, 1.0, 0.05, 0.2)
        
        call, _ = mc.european_call(n_sims=50000)
        put, _ = mc.european_put(n_sims=50000)
        
        # Put-call parity
        parity = call - 100 + 100 * np.exp(-0.05)
        assert np.isclose(put, parity, rtol=0.02)
    
    def test_asian_option(self):
        """Test Asian option pricing."""
        mc = MonteCarloOption(100, 100, 1.0, 0.05, 0.2)
        
        asian_call, _ = mc.asian_call(n_sims=30000)
        european_call, _ = mc.european_call(n_sims=30000)
        
        # Asian call should be cheaper (averaging reduces volatility)
        assert asian_call < european_call


class TestGreeksCalculator:
    """Test Greeks calculator."""
    
    def test_numerical_greeks(self):
        """Test numerical Greeks against analytical."""
        bs = BlackScholes(100, 100, 1.0, 0.05, 0.2)
        calc = GreeksCalculator(bs)
        
        # Compare numerical and analytical delta
        numerical_delta = calc.delta('call')
        analytical_delta = bs.delta('call')
        assert np.isclose(numerical_delta, analytical_delta, rtol=1e-3)
        
        # Compare gamma
        numerical_gamma = calc.gamma()
        analytical_gamma = bs.gamma()
        assert np.isclose(numerical_gamma, analytical_gamma, rtol=1e-2)


class TestImpliedVolatility:
    """Test implied volatility calculations."""
    
    def test_implied_vol_recovery(self):
        """Test that we can recover input volatility."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        bs = BlackScholes(S, K, T, r, sigma)
        market_price = bs.call_price()
        
        iv_calc = ImpliedVolatility(S, r)
        implied = iv_calc.calculate(K, T, market_price, 'call')
        
        assert np.isclose(implied, sigma, rtol=1e-4)
    
    def test_volatility_smile(self):
        """Test implied vol for different strikes."""
        S, T, r = 100, 1.0, 0.05
        
        iv_calc = ImpliedVolatility(S, r)
        
        # Create prices with smile
        strikes = [90, 95, 100, 105, 110]
        vols = [0.25, 0.22, 0.20, 0.22, 0.25]
        
        implied_vols = []
        for K, vol in zip(strikes, vols):
            bs = BlackScholes(S, K, T, r, vol)
            price = bs.call_price()
            implied = iv_calc.calculate(K, T, price, 'call')
            implied_vols.append(implied)
        
        # Should recover approximately the same smile
        for i, (expected, actual) in enumerate(zip(vols, implied_vols)):
            assert np.isclose(expected, actual, rtol=1e-3)


class TestOptionsStrategies:
    """Test options strategies."""
    
    def test_bull_call_spread(self):
        """Test bull call spread."""
        spread = SpreadStrategy.bull_call_spread(
            S=100, K1=95, K2=105, T=1.0, r=0.05, sigma=0.2
        )
        
        # Price should be positive
        assert spread.price() > 0
        
        # Max profit should be positive
        assert spread.max_profit() > 0
        
        # Max loss should be negative (cost of spread)
        assert spread.max_loss() < 0
        
        # Payoff at K2 should be max profit
        payoff_at_max = spread.payoff(np.array([105]))
        assert np.isclose(payoff_at_max[0], spread.max_profit(), rtol=1e-3)
    
    def test_straddle(self):
        """Test straddle."""
        straddle = SpreadStrategy.straddle(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2
        )
        
        # Price should be sum of call and put
        bs = BlackScholes(100, 100, 1.0, 0.05, 0.2)
        expected_price = bs.call_price() + bs.put_price()
        assert np.isclose(straddle.price(), expected_price, rtol=1e-3)
        
        # Max profit at extreme moves
        assert straddle.max_profit() > 50  # Large potential profit
    
    def test_iron_condor(self):
        """Test iron condor."""
        condor = SpreadStrategy.iron_condor(
            S=100, K1=90, K2=95, K3=105, K4=110, T=1.0, r=0.05, sigma=0.2
        )
        
        # Should receive a credit (negative price)
        assert condor.price() < 0
        
        # Max profit at middle range
        payoff_middle = condor.payoff(np.array([100]))
        assert payoff_middle[0] > 0
        
        # Limited risk
        assert abs(condor.max_loss()) < 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])