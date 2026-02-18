"""
Example 1: Options Pricing with QuantFin

This example demonstrates:
1. Black-Scholes pricing
2. Greeks calculation
3. Binomial and Monte Carlo models
4. Implied volatility calculation
5. Options strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from quantfin.options.models import BlackScholes, BinomialModel, MonteCarloOption
from quantfin.options.greeks import GreeksCalculator
from quantfin.options.implied_vol import ImpliedVolatility
from quantfin.options.strategies import SpreadStrategy

# =============================================================================
# Part 1: Black-Scholes Pricing
# =============================================================================

print("=" * 60)
print("PART 1: Black-Scholes Option Pricing")
print("=" * 60)

# Parameters
S = 100  # Stock price
K = 100  # Strike
T = 1.0  # 1 year to expiration
r = 0.05  # 5% risk-free rate
sigma = 0.2  # 20% volatility

# Create Black-Scholes model
bs = BlackScholes(S, K, T, r, sigma)

# Price options
call_price = bs.call_price()
put_price = bs.put_price()

print(f"\nParameters:")
print(f"  Stock Price (S): ${S}")
print(f"  Strike Price (K): ${K}")
print(f"  Time to Expiration (T): {T} years")
print(f"  Risk-free Rate (r): {r:.1%}")
print(f"  Volatility (σ): {sigma:.1%}")

print(f"\nOption Prices:")
print(f"  European Call: ${call_price:.4f}")
print(f"  European Put: ${put_price:.4f}")

# =============================================================================
# Part 2: Greeks Calculation
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Option Greeks")
print("=" * 60)

# Calculate Greeks
greeks = bs.all_greeks('call')

print(f"\nCall Option Greeks:")
print(f"  Delta: {greeks['delta']:.4f}")
print(f"  Gamma: {greeks['gamma']:.4f}")
print(f"  Vega: {greeks['vega']:.4f} (per 1% vol change)")
print(f"  Theta: {greeks['theta']:.4f} (daily)")
print(f"  Rho: {greeks['rho']:.4f} (per 1% rate change)")
print(f"  Vanna: {greeks['vanna']:.4f}")
print(f"  Charm: {greeks['charm']:.4f}")

# Put Greeks
put_greeks = bs.all_greeks('put')
print(f"\nPut Option Greeks:")
print(f"  Delta: {put_greeks['delta']:.4f}")
print(f"  Gamma: {put_greeks['gamma']:.4f}")

# =============================================================================
# Part 3: Model Comparison
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Model Comparison")
print("=" * 60)

# Binomial model
binomial = BinomialModel(S, K, T, r, sigma, n=100)
bin_call = binomial.call_price()
bin_put = binomial.put_price(american=True)

# Monte Carlo
mc = MonteCarloOption(S, K, T, r, sigma)
mc_call, mc_err = mc.european_call(n_sims=100000)
mc_put, _ = mc.european_put(n_sims=100000)

print(f"\nCall Option Price Comparison:")
print(f"  Black-Scholes: ${call_price:.4f}")
print(f"  Binomial (100 steps): ${bin_call:.4f}")
print(f"  Monte Carlo: ${mc_call:.4f} ± {mc_err:.4f}")

print(f"\nPut Option Price Comparison:")
print(f"  Black-Scholes: ${put_price:.4f}")
print(f"  Binomial American: ${bin_put:.4f}")
print(f"  Monte Carlo: ${mc_put:.4f}")

# =============================================================================
# Part 4: Implied Volatility
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Implied Volatility")
print("=" * 60)

# Calculate implied vol from market price
market_price = call_price + 0.5  # Simulated market price

iv_calc = ImpliedVolatility(S, r)
implied_vol = iv_calc.calculate(K, T, market_price, 'call')

print(f"\nImplied Volatility Calculation:")
print(f"  Market Price: ${market_price:.4f}")
print(f"  Implied Volatility: {implied_vol:.2%}")
print(f"  Input Volatility: {sigma:.2%}")

# =============================================================================
# Part 5: Options Strategies
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Options Strategies")
print("=" * 60)

# Bull Call Spread
K1 = 95  # Long call strike
K2 = 105  # Short call strike

spread = SpreadStrategy.bull_call_spread(S, K1, K2, T, r, sigma)
spread_price = spread.price()
breakevens = spread.breakevens()
max_profit = spread.max_profit()
max_loss = spread.max_loss()

print(f"\nBull Call Spread (K1=${K1}, K2=${K2}):")
print(f"  Strategy Price: ${spread_price:.4f}")
print(f"  Max Profit: ${max_profit:.4f}")
print(f"  Max Loss: ${max_loss:.4f}")
print(f"  Breakeven(s): ${breakevens[0]:.2f}" if breakevens else "  No breakeven")

# Iron Condor
K1, K2, K3, K4 = 90, 95, 105, 110
condor = SpreadStrategy.iron_condor(S, K1, K2, K3, K4, T, r, sigma)
condor_price = condor.price()

print(f"\nIron Condor (K1=${K1}, K2=${K2}, K3=${K3}, K4=${K4}):")
print(f"  Strategy Price (credit): ${-condor_price:.4f}")
print(f"  Max Profit: ${condor.max_profit():.4f}")
print(f"  Max Loss: ${abs(condor.max_loss()):.4f}")

# =============================================================================
# Part 6: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Creating Visualizations")
print("=" * 60)

# Plot payoff diagram
spot_range = np.linspace(80, 120, 100)
payoff = spread.payoff(spot_range)
pnl = spread.pnl(spot_range)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(spot_range, payoff, 'b-', linewidth=2, label='Payoff')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=S, color='r', linestyle='--', alpha=0.5, label='Current Spot')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Payoff ($)')
plt.title('Bull Call Spread Payoff')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(spot_range, pnl, 'g-', linewidth=2, label='P&L')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=S, color='r', linestyle='--', alpha=0.5, label='Current Spot')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('P&L ($)')
plt.title('Bull Call Spread P&L')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('options_strategy_payoff.png', dpi=150)
print("\nSaved visualization to 'options_strategy_payoff.png'")

# =============================================================================
# Part 7: Exotic Options
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Exotic Options")
print("=" * 60)

# Asian option
asian_call, asian_err = mc.asian_call(n_sims=50000)
asian_put, _ = mc.asian_put(n_sims=50000)

print(f"\nAsian Options (arithmetic average):")
print(f"  Asian Call: ${asian_call:.4f} ± {asian_err:.4f}")
print(f"  Asian Put: ${asian_put:.4f}")

# Lookback option
lookback_call, lookback_err = mc.lookback_call(n_sims=50000)
lookback_put, _ = mc.lookback_put(n_sims=50000)

print(f"\nLookback Options (fixed strike):")
print(f"  Lookback Call: ${lookback_call:.4f} ± {lookback_err:.4f}")
print(f"  Lookback Put: ${lookback_put:.4f}")

# Barrier option
barrier = 110
barrier_call, barrier_err = mc.barrier_call(barrier, 'up-and-out', n_sims=50000)

print(f"\nBarrier Options (barrier=${barrier}):")
print(f"  Up-and-Out Call: ${barrier_call:.4f} ± {barrier_err:.4f}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)