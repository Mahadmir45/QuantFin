"""
Example 3: Topology Alpha Strategy

This example demonstrates the Topology Alpha Model which uses:
1. Topological Data Analysis (TDA) on correlation graphs
2. Persistent homology (Betti numbers)
3. Laplacian diffusion for residual alpha
4. Risk-parity weighting
5. Regime-based exposure timing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantfin.core.data import SyntheticDataProvider
from quantfin.strategies.topology import TopologyAlphaStrategy
from quantfin.core.utils import sharpe_ratio, maximum_drawdown

# =============================================================================
# Part 1: Generate Market Data
# =============================================================================

print("=" * 60)
print("PART 1: Generating Market Data")
print("=" * 60)

# Create synthetic data provider with correlation
provider = SyntheticDataProvider()

# Stock universe
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
          'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD', 'UBER', 'LYFT']

# Generate correlated prices
np.random.seed(42)
prices = provider.get_prices(stocks, '2020-01-01', '2023-12-31')

# Generate synthetic VIX
vix = pd.Series(
    20 + 10 * np.random.randn(len(prices)).cumsum() * 0.1,
    index=prices.index
).clip(10, 60)

print(f"\nGenerated data:")
print(f"  Stocks: {len(stocks)}")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"  Observations: {len(prices)}")
print(f"  VIX range: {vix.min():.1f} - {vix.max():.1f}")

# =============================================================================
# Part 2: Create Topology Alpha Strategy
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Creating Topology Alpha Strategy")
print("=" * 60)

# Initialize strategy
strategy = TopologyAlphaStrategy(
    stocks=stocks,
    window_size=90,
    vix_threshold=35,
    betti_threshold_low=30,
    betti_threshold_high=45,
    diffusion_time=0.5,
    use_ml=False
)

print(f"\nStrategy Parameters:")
print(f"  Window Size: {strategy.window_size} days")
print(f"  VIX Threshold: {strategy.vix_threshold}")
print(f"  Betti Low Threshold: {strategy.betti_threshold_low}")
print(f"  Betti High Threshold: {strategy.betti_threshold_high}")
print(f"  Diffusion Time: {strategy.diffusion_time}")

# =============================================================================
# Part 3: Run Backtest
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Running Backtest")
print("=" * 60)

# Run backtest
results = strategy.backtest(
    prices=prices,
    vix=vix,
    transaction_cost=0.0005,
    initial_capital=100000.0
)

print(f"\nBacktest Results:")
print(f"  Total Return: {results['metrics']['total_return']:.2%}")
print(f"  Annual Return: {results['metrics']['annual_return']:.2%}")
print(f"  Annual Volatility: {results['metrics']['annual_volatility']:.2%}")
print(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
print(f"  Calmar Ratio: {results['metrics']['calmar_ratio']:.3f}")
print(f"  Average Turnover: {results['metrics']['avg_turnover']:.2%}")
print(f"  Average Exposure: {results['metrics']['avg_exposure']:.2f}x")

# =============================================================================
# Part 4: Analyze Signals
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Signal Analysis")
print("=" * 60)

signals_df = results['signals']

# Regime distribution
regime_counts = signals_df['regime'].value_counts()
print(f"\nRegime Distribution:")
for regime, count in regime_counts.items():
    pct = count / len(signals_df) * 100
    print(f"  {regime}: {count} days ({pct:.1f}%)")

# Exposure by regime
print(f"\nAverage Exposure by Regime:")
exposure_by_regime = signals_df.groupby('regime')['exposure'].mean()
for regime, exp in exposure_by_regime.items():
    print(f"  {regime}: {exp:.2f}x")

# Betti statistics
print(f"\nBetti Statistics:")
print(f"  Mean: {signals_df['betti_std'].mean():.2f}")
print(f"  Std: {signals_df['betti_std'].std():.2f}")
print(f"  Min: {signals_df['betti_std'].min():.2f}")
print(f"  Max: {signals_df['betti_std'].max():.2f}")

# =============================================================================
# Part 5: Comparison with Buy-and-Hold
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Comparison with Buy-and-Hold")
print("=" * 60)

# Calculate equal-weight benchmark
returns = prices.pct_change().dropna()
benchmark_returns = returns.mean(axis=1)
benchmark_equity = 100000 * (1 + benchmark_returns).cumprod()

# Calculate metrics
bench_sharpe = sharpe_ratio(benchmark_returns)
bench_maxdd = maximum_drawdown(benchmark_equity / 100000)

print(f"\nBuy-and-Hold Benchmark (Equal Weight):")
print(f"  Total Return: {(benchmark_equity.iloc[-1] / 100000 - 1):.2%}")
print(f"  Annual Return: {benchmark_returns.mean() * 252:.2%}")
print(f"  Annual Volatility: {benchmark_returns.std() * np.sqrt(252):.2%}")
print(f"  Sharpe Ratio: {bench_sharpe:.3f}")
print(f"  Max Drawdown: {bench_maxdd:.2%}")

print(f"\nStrategy vs Benchmark:")
print(f"  Return Difference: {(results['metrics']['total_return'] - (benchmark_equity.iloc[-1] / 100000 - 1)):.2%}")
print(f"  Sharpe Difference: {results['metrics']['sharpe_ratio'] - bench_sharpe:.3f}")
print(f"  Drawdown Improvement: {bench_maxdd - results['metrics']['max_drawdown']:.2%}")

# =============================================================================
# Part 6: Visualizations
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Creating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1. Equity Curves
ax = axes[0, 0]
ax.plot(results['equity_curve'] / 100000, label='Topology Alpha', linewidth=2)
ax.plot(benchmark_equity / 100000, label='Buy & Hold', linewidth=2, alpha=0.7)
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('Equity Curve Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Drawdowns
ax = axes[0, 1]
strategy_dd = (results['equity_curve'] / results['equity_curve'].expanding().max() - 1) * 100
bench_dd = (benchmark_equity / benchmark_equity.expanding().max() - 1) * 100
ax.fill_between(strategy_dd.index, strategy_dd, 0, alpha=0.5, label='Topology Alpha')
ax.fill_between(bench_dd.index, bench_dd, 0, alpha=0.5, label='Buy & Hold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Regime and Exposure
ax = axes[1, 0]
ax2 = ax.twinx()

# Plot exposure
ax.plot(signals_df.index, signals_df['exposure'], 'b-', linewidth=2, label='Exposure')
ax.set_ylabel('Exposure', color='b')
ax.tick_params(axis='y', labelcolor='b')

# Plot VIX
ax2.plot(signals_df.index, signals_df['vix'], 'r-', alpha=0.5, label='VIX')
ax2.set_ylabel('VIX', color='r')
ax2.tick_params(axis='y', labelcolor='r')

ax.set_xlabel('Date')
ax.set_title('Exposure and VIX Over Time')
ax.grid(True, alpha=0.3)

# 4. Betti Numbers
ax = axes[1, 1]
ax.plot(signals_df.index, signals_df['betti_std'], 'g-', linewidth=2)
ax.axhline(y=strategy.betti_threshold_low, color='green', linestyle='--', 
          alpha=0.5, label='Low Risk Threshold')
ax.axhline(y=strategy.betti_threshold_high, color='red', linestyle='--', 
          alpha=0.5, label='High Risk Threshold')
ax.set_xlabel('Date')
ax.set_ylabel('Betti Std')
ax.set_title('Betti Number Standard Deviation')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Rolling Sharpe
ax = axes[2, 0]
window = 63
strategy_rolling = results['returns'].rolling(window).mean() * 252 / (results['returns'].rolling(window).std() * np.sqrt(252))
bench_rolling = benchmark_returns.rolling(window).mean() * 252 / (benchmark_returns.rolling(window).std() * np.sqrt(252))
ax.plot(strategy_rolling, label='Topology Alpha', linewidth=2)
ax.plot(bench_rolling, label='Buy & Hold', linewidth=2, alpha=0.7)
ax.set_xlabel('Date')
ax.set_ylabel('Rolling Sharpe (63-day)')
ax.set_title('Rolling Sharpe Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Monthly Returns Heatmap
ax = axes[2, 1]
monthly_returns = results['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_df = pd.DataFrame({
    'Year': monthly_returns.index.year,
    'Month': monthly_returns.index.month,
    'Return': monthly_returns.values
})
pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
ax.set_xticks(range(12))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_title('Monthly Returns Heatmap (%)')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('topology_alpha_strategy.png', dpi=150)
print("\nSaved visualization to 'topology_alpha_strategy.png'")

# =============================================================================
# Part 7: Weight Analysis
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Portfolio Weight Analysis")
print("=" * 60)

weights_df = results['weights']

# Average weights
avg_weights = weights_df.mean().sort_values(ascending=False)
print(f"\nAverage Portfolio Weights (Top 10):")
for stock, weight in avg_weights.head(10).items():
    print(f"  {stock}: {weight:.2%}")

# Weight statistics
print(f"\nWeight Statistics:")
print(f"  Mean max weight: {weights_df.max(axis=1).mean():.2%}")
print(f"  Mean min weight: {weights_df.min(axis=1).mean():.2%}")
print(f"  Mean concentration (HHI): {(weights_df ** 2).sum(axis=1).mean():.3f}")

# =============================================================================
# Part 8: Summary
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: Summary")
print("=" * 60)

print(f"""
Topology Alpha Strategy Summary:
================================

Performance:
  - Total Return: {results['metrics']['total_return']:.2%}
  - Annual Return: {results['metrics']['annual_return']:.2%}
  - Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}
  - Max Drawdown: {results['metrics']['max_drawdown']:.2%}
  - Calmar Ratio: {results['metrics']['calmar_ratio']:.3f}

Risk Management:
  - Average Exposure: {results['metrics']['avg_exposure']:.2f}x
  - Average Turnover: {results['metrics']['avg_turnover']:.2%}
  - Time in Low Risk Regime: {(signals_df['regime'] == 'low_risk').mean():.1%}
  - Time in High Risk Regime: {(signals_df['regime'] == 'high_risk').mean():.1%}

Key Insights:
  1. Uses topological data analysis to detect market regimes
  2. Adjusts exposure based on VIX and Betti numbers
  3. Employs risk-parity weighting for diversification
  4. Laplacian diffusion identifies local vs global mispricings
""")

print("=" * 60)
print("Example completed successfully!")
print("=" * 60)