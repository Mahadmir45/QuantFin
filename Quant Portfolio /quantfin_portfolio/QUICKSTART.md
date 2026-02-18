# QuantFin Pro - Quick Start Guide

Get started with QuantFin Pro in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/Mahadmir45/QuantFin.git
cd QuantFin

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Your First Script

### 1. Price an Option

```python
from quantfin.options.models import BlackScholes

# Create Black-Scholes model
bs = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

# Price call option
price = bs.call_price()
print(f"Call option price: ${price:.2f}")

# Calculate Greeks
greeks = bs.all_greeks('call')
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
```

### 2. Optimize a Portfolio

```python
import pandas as pd
from quantfin.portfolio.optimization import PortfolioOptimizer

# Load returns data
returns = pd.read_csv('returns.csv', index_col=0)

# Create optimizer
optimizer = PortfolioOptimizer(returns)

# Maximize Sharpe ratio
weights, info = optimizer.optimize_max_sharpe()

print(f"Expected return: {info['return']:.2%}")
print(f"Volatility: {info['risk']:.2%}")
print(f"Sharpe ratio: {info['sharpe']:.3f}")
```

### 3. Run a Strategy Backtest

```python
from quantfin.strategies.topology import TopologyAlphaStrategy

# Create strategy
strategy = TopologyAlphaStrategy(
    stocks=['AAPL', 'MSFT', 'GOOGL'],
    window_size=90,
    vix_threshold=35
)

# Run backtest
results = strategy.backtest(
    prices=prices_df,
    vix=vix_series,
    transaction_cost=0.0005
)

print(f"Total return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.3f}")
```

## Common Tasks

### Fetch Market Data

```python
from quantfin.core.data import YahooFinanceProvider

provider = YahooFinanceProvider()
prices = provider.get_prices(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)
returns = prices.pct_change().dropna()
```

### Calculate Risk Metrics

```python
from quantfin.portfolio.risk import RiskMetrics

risk = RiskMetrics(returns)

# Value at Risk
var_95 = risk.var_historical(confidence=0.95)
print(f"VaR (95%): {var_95:.2%}")

# Maximum drawdown
max_dd = risk.maximum_drawdown()
print(f"Max drawdown: {max_dd:.2%}")

# Full summary
summary = risk.summary()
print(summary)
```

### Build Options Strategies

```python
from quantfin.options.strategies import SpreadStrategy

# Bull call spread
spread = SpreadStrategy.bull_call_spread(
    S=100, K1=95, K2=105, T=1.0, r=0.05, sigma=0.2
)

print(f"Strategy price: ${spread.price():.2f}")
print(f"Max profit: ${spread.max_profit():.2f}")
print(f"Max loss: ${spread.max_loss():.2f}")
```

## Next Steps

1. **Read the Documentation**: See `DOCUMENTATION.md` for detailed API reference
2. **Run Examples**: Check `examples/` directory for complete scripts
3. **Explore Modules**: Each module has comprehensive docstrings
4. **Build Your Own**: Extend the base classes for custom strategies

## Getting Help

- Check the documentation
- Review the examples
- Open an issue on GitHub
- Join the community discussions

Happy quanting! 🚀