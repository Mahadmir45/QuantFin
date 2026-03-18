"""
Example 2: Portfolio Optimization with QuantFin

This example demonstrates:
1. Mean-variance optimization
2. Risk parity
3. CVaR optimization
4. Efficient frontier
5. Backtesting strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantfin.core.data import SyntheticDataProvider
from quantfin.portfolio.optimization import PortfolioOptimizer
from quantfin.portfolio.risk import RiskMetrics, RiskParity
from quantfin.portfolio.backtest import StrategyBacktest, BenchmarkComparison

# =============================================================================
# Part 1: Generate Synthetic Data
# =============================================================================

print("=" * 60)
print("PART 1: Generating Synthetic Market Data")
print("=" * 60)

# Create synthetic data provider
provider = SyntheticDataProvider()

# Generate price data for 10 assets
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
           'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']

prices = provider.get_prices(tickers, '2020-01-01', '2023-12-31', seed=42)
returns = prices.pct_change().dropna()

print(f"\nGenerated data for {len(tickers)} assets:")
print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"  Total observations: {len(returns)}")
print(f"\nAsset statistics:")
print(returns.mean() * 252)  # Annualized returns

# =============================================================================
# Part 2: Mean-Variance Optimization
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Mean-Variance Optimization")
print("=" * 60)

# Create optimizer
optimizer = PortfolioOptimizer(returns)

# Maximize Sharpe ratio
weights_sharpe, info_sharpe = optimizer.optimize_max_sharpe()

print(f"\nMaximum Sharpe Portfolio:")
print(f"  Expected Return: {info_sharpe['return']:.2%}")
print(f"  Volatility: {info_sharpe['risk']:.2%}")
print(f"  Sharpe Ratio: {info_sharpe['sharpe']:.3f}")
print(f"\nWeights:")
weights_df = optimizer.get_weights_df(weights_sharpe)
print(weights_df[weights_df['weight'] > 0.01])

# Minimum variance portfolio
weights_minvar, info_minvar = optimizer.optimize_min_variance()

print(f"\nMinimum Variance Portfolio:")
print(f"  Expected Return: {info_minvar['return']:.2%}")
print(f"  Volatility: {info_minvar['risk']:.2%}")
print(f"  Sharpe Ratio: {info_minvar['sharpe']:.3f}")

# =============================================================================
# Part 3: Risk Parity
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Risk Parity Portfolio")
print("=" * 60)

# Create risk parity portfolio
rp = RiskParity(returns)
rp_weights = rp.get_weights()

# Get risk contributions
rc_df = rp.get_risk_contributions(rp_weights)

print(f"\nRisk Parity Portfolio:")
print(rc_df.to_string(index=False))

# =============================================================================
# Part 4: CVaR Optimization
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: CVaR Optimization")
print("=" * 60)

# Optimize with CVaR constraint
weights_cvar, info_cvar = optimizer.optimize_cvar(
    confidence=0.95,
    target_return=0.12
)

print(f"\nCVaR-Optimized Portfolio:")
print(f"  Expected Return: {info_cvar['return']:.2%}")
print(f"  CVaR (95%): {info_cvar['cvar']:.2%}")
print(f"\nWeights:")
print(optimizer.get_weights_df(weights_cvar)[weights_cvar > 0.01])

# =============================================================================
# Part 5: Efficient Frontier
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Efficient Frontier")
print("=" * 60)

# Generate efficient frontier
frontier = optimizer.efficient_frontier(n_points=50)

print(f"\nEfficient Frontier generated with {len(frontier)} points")
print(f"  Min Return: {frontier['return'].min():.2%}")
print(f"  Max Return: {frontier['return'].max():.2%}")
print(f"  Min Risk: {frontier['risk'].min():.2%}")
print(f"  Max Risk: {frontier['risk'].max():.2%}")

# Plot efficient frontier
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(frontier['risk'] * 100, frontier['return'] * 100, 'b-', linewidth=2, label='Efficient Frontier')
plt.scatter(info_minvar['risk'] * 100, info_minvar['return'] * 100, 
           c='green', s=100, marker='*', label='Min Variance', zorder=5)
plt.scatter(info_sharpe['risk'] * 100, info_sharpe['return'] * 100, 
           c='red', s=100, marker='*', label='Max Sharpe', zorder=5)
plt.xlabel('Risk (Volatility %)')
plt.ylabel('Expected Return (%)')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot portfolio compositions
plt.subplot(1, 2, 2)
portfolios = pd.DataFrame({
    'Max Sharpe': weights_sharpe,
    'Min Variance': weights_minvar,
    'Risk Parity': rp_weights,
    'CVaR': weights_cvar
}, index=tickers)

portfolios.plot(kind='bar', ax=plt.gca())
plt.title('Portfolio Weights Comparison')
plt.xlabel('Assets')
plt.ylabel('Weight')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('portfolio_optimization.png', dpi=150)
print("\nSaved visualization to 'portfolio_optimization.png'")

# =============================================================================
# Part 6: Risk Analysis
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Risk Analysis")
print("=" * 60)

# Calculate risk metrics for each portfolio
portfolios_dict = {
    'Max Sharpe': weights_sharpe,
    'Min Variance': weights_minvar,
    'Risk Parity': rp_weights,
    'CVaR': weights_cvar
}

risk_results = []
for name, weights in portfolios_dict.items():
    risk_metrics = RiskMetrics(returns)
    summary = risk_metrics.summary(weights)
    summary['Portfolio'] = name
    risk_results.append(summary)

risk_comparison = pd.concat(risk_results, axis=1)
print("\nRisk Metrics Comparison:")
print(risk_comparison.round(4))

# =============================================================================
# Part 7: Backtesting
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Strategy Backtesting")
print("=" * 60)

# Create backtest engine
backtest = StrategyBacktest(returns, rebalance_freq='M', transaction_cost=0.001)

# Run different strategies
print("\nRunning backtests...")

# Equal weight
result_ew = backtest.equal_weight()
print(f"\nEqual Weight:")
print(f"  Total Return: {result_ew.metrics['total_return']:.2%}")
print(f"  Sharpe Ratio: {result_ew.metrics['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {result_ew.metrics['max_drawdown']:.2%}")

# Inverse volatility
result_iv = backtest.inverse_volatility(window=63)
print(f"\nInverse Volatility:")
print(f"  Total Return: {result_iv.metrics['total_return']:.2%}")
print(f"  Sharpe Ratio: {result_iv.metrics['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {result_iv.metrics['max_drawdown']:.2%}")

# Risk parity
result_rp = backtest.risk_parity(window=63)
print(f"\nRisk Parity:")
print(f"  Total Return: {result_rp.metrics['total_return']:.2%}")
print(f"  Sharpe Ratio: {result_rp.metrics['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {result_rp.metrics['max_drawdown']:.2%}")

# =============================================================================
# Part 8: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: Backtest Visualization")
print("=" * 60)

plt.figure(figsize=(14, 10))

# Equity curves
plt.subplot(2, 2, 1)
plt.plot(result_ew.equity_curve / result_ew.equity_curve.iloc[0], 
         label='Equal Weight', linewidth=2)
plt.plot(result_iv.equity_curve / result_iv.equity_curve.iloc[0], 
         label='Inverse Vol', linewidth=2)
plt.plot(result_rp.equity_curve / result_rp.equity_curve.iloc[0], 
         label='Risk Parity', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Normalized Equity')
plt.title('Strategy Performance Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Drawdowns
plt.subplot(2, 2, 2)
for result, name in [(result_ew, 'EW'), (result_iv, 'IV'), (result_rp, 'RP')]:
    equity = result.equity_curve
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    plt.plot(drawdown * 100, label=name, linewidth=2)

plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.title('Strategy Drawdowns')
plt.legend()
plt.grid(True, alpha=0.3)

# Rolling Sharpe
plt.subplot(2, 2, 3)
window = 63
for result, name in [(result_ew, 'EW'), (result_iv, 'IV'), (result_rp, 'RP')]:
    rolling_ret = result.returns.rolling(window).mean() * 252
    rolling_vol = result.returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_ret / rolling_vol
    plt.plot(rolling_sharpe, label=name, linewidth=2)

plt.xlabel('Date')
plt.ylabel('Rolling Sharpe (63-day)')
plt.title('Rolling Sharpe Ratio')
plt.legend()
plt.grid(True, alpha=0.3)

# Performance metrics bar chart
plt.subplot(2, 2, 4)
metrics = ['total_return', 'sharpe_ratio', 'calmar_ratio']
strategies = ['EW', 'IV', 'RP']
data = {
    'EW': [result_ew.metrics[m] for m in metrics],
    'IV': [result_iv.metrics[m] for m in metrics],
    'RP': [result_rp.metrics[m] for m in metrics]
}

x = np.arange(len(metrics))
width = 0.25

for i, strategy in enumerate(strategies):
    plt.bar(x + i * width, data[strategy], width, label=strategy)

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics Comparison')
plt.xticks(x + width, ['Total Return', 'Sharpe', 'Calmar'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('portfolio_backtest.png', dpi=150)
print("\nSaved visualization to 'portfolio_backtest.png'")

# =============================================================================
# Part 9: Summary
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: Summary")
print("=" * 60)

summary = pd.DataFrame({
    'Equal Weight': result_ew.metrics,
    'Inverse Vol': result_iv.metrics,
    'Risk Parity': result_rp.metrics
}).T

print("\nBacktest Results Summary:")
print(summary[['total_return', 'annual_return', 'annual_volatility', 
               'sharpe_ratio', 'max_drawdown', 'calmar_ratio']].round(4))

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)