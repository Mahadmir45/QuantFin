# QuantFin Pro - Comprehensive Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Module](#core-module)
3. [Options Module](#options-module)
4. [Portfolio Module](#portfolio-module)
5. [Strategies Module](#strategies-module)
6. [Backtesting Module](#backtesting-module)
7. [Examples](#examples)
8. [API Reference](#api-reference)

---

## Architecture Overview

QuantFin Pro follows a modular architecture designed for extensibility and maintainability:

```
quantfin/
├── core/           # Shared infrastructure
├── options/        # Options pricing & Greeks
├── portfolio/      # Portfolio optimization & risk
├── strategies/     # Trading strategies
├── backtesting/    # Backtesting engine
└── tests/          # Unit tests
```

### Design Principles

1. **Modularity**: Each module is self-contained with clear interfaces
2. **Extensibility**: Easy to add new models, strategies, and features
3. **Type Safety**: Type hints throughout for better IDE support
4. **Documentation**: Comprehensive docstrings and examples

---

## Core Module

### Configuration

```python
from quantfin.core.config import Config

# Create configuration
config = Config(
    data_dir='./data',
    cache_dir='./cache',
    risk_free_rate=0.045,
    default_commission=0.001
)

# Load from file
config = Config.from_json('config.json')
```

### Data Providers

```python
from quantfin.core.data import YahooFinanceProvider, PolygonProvider

# Yahoo Finance
yahoo = YahooFinanceProvider()
prices = yahoo.get_prices(['AAPL', 'MSFT'], '2020-01-01', '2023-12-31')

# Polygon.io
polygon = PolygonProvider(api_key='your_key')
prices = polygon.get_prices(['AAPL'], '2020-01-01', '2023-12-31')
```

### Utility Functions

```python
from quantfin.core.utils import (
    sharpe_ratio, maximum_drawdown, 
    var_historical, cvar_historical
)

# Calculate metrics
sharpe = sharpe_ratio(returns, risk_free_rate=0.045)
max_dd = maximum_drawdown(equity_curve)
var_95 = var_historical(returns, confidence=0.95)
```

---

## Options Module

### Black-Scholes Model

```python
from quantfin.options.models import BlackScholes

# Create model
bs = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

# Price options
call_price = bs.call_price()
put_price = bs.put_price()

# Calculate Greeks
greeks = bs.all_greeks('call')
print(f"Delta: {greeks['delta']}")
print(f"Gamma: {greeks['gamma']}")
print(f"Vega: {greeks['vega']}")
```

### Binomial Model

```python
from quantfin.options.models import BinomialModel

# Create model
binomial = BinomialModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2, n=100)

# Price European and American options
european_call = binomial.call_price(american=False)
american_put = binomial.put_price(american=True)
```

### Monte Carlo

```python
from quantfin.options.models import MonteCarloOption

# Create model
mc = MonteCarloOption(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

# Price exotic options
european_call, std_err = mc.european_call(n_sims=100000)
asian_call, _ = mc.asian_call(n_sims=50000)
lookback_call, _ = mc.lookback_call(n_sims=50000)
barrier_call, _ = mc.barrier_call(barrier=110, barrier_type='up-and-out')
```

### Implied Volatility

```python
from quantfin.options.implied_vol import ImpliedVolatility, IVSurface

# Calculate implied vol
iv_calc = ImpliedVolatility(S=100, r=0.05)
implied_vol = iv_calc.calculate(K=100, T=1.0, market_price=10.5, option_type='call')

# Build IV surface
strikes = np.linspace(80, 120, 20)
maturities = np.array([0.25, 0.5, 1.0, 2.0])
market_prices = np.array([[...]])  # Shape: (len(strikes), len(maturities))

iv_surface = iv_calc.calculate_batch(strikes, maturities, market_prices)
surface = IVSurface(strikes, maturities, iv_surface)
```

### Options Strategies

```python
from quantfin.options.strategies import SpreadStrategy

# Bull call spread
spread = SpreadStrategy.bull_call_spread(S=100, K1=95, K2=105, T=1.0, r=0.05, sigma=0.2)
price = spread.price()
payoff = spread.payoff(spot_prices=np.linspace(80, 120, 100))
breakevens = spread.breakevens()

# Other strategies
iron_condor = SpreadStrategy.iron_condor(S=100, K1=90, K2=95, K3=105, K4=110, T=1.0, r=0.05, sigma=0.2)
straddle = SpreadStrategy.straddle(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
```

---

## Portfolio Module

### Portfolio Optimization

```python
from quantfin.portfolio.optimization import PortfolioOptimizer

# Create optimizer
optimizer = PortfolioOptimizer(returns_df)

# Mean-variance optimization
weights, info = optimizer.optimize_mean_variance(target_return=0.12)

# Max Sharpe
weights, info = optimizer.optimize_max_sharpe()

# Min variance
weights, info = optimizer.optimize_min_variance()

# CVaR optimization
weights, info = optimizer.optimize_cvar(confidence=0.95, target_return=0.12)

# Risk parity
weights, info = optimizer.optimize_risk_parity()

# Efficient frontier
frontier = optimizer.efficient_frontier(n_points=50)
```

### Risk Metrics

```python
from quantfin.portfolio.risk import RiskMetrics, RiskParity

# Calculate risk metrics
risk = RiskMetrics(returns_df)

# Various VaR methods
var_hist = risk.var_historical(confidence=0.95, weights=portfolio_weights)
var_param = risk.var_parametric(confidence=0.95, weights=portfolio_weights)

# CVaR
cvar = risk.cvar(confidence=0.95, weights=portfolio_weights)

# Drawdown
max_dd = risk.maximum_drawdown(weights=portfolio_weights)
dd_series = risk.drawdown_series(weights=portfolio_weights)

# Full summary
summary = risk.summary(weights=portfolio_weights, benchmark_returns=spy_returns)

# Risk parity
rp = RiskParity(returns_df)
weights = rp.get_weights()
rc_df = rp.get_risk_contributions(weights)
```

### Performance Attribution

```python
from quantfin.portfolio.attribution import PerformanceAttribution

# Create attribution analysis
attribution = PerformanceAttribution(
    port_returns=port_returns,
    port_weights=port_weights,
    bench_returns=bench_returns,
    bench_weights=bench_weights
)

# Brinson attribution
brinson = attribution.brinson_attribution()
summary = attribution.summary()

# Factor attribution
factor_attr = attribution.factor_attribution(factor_returns, factor_exposures)
```

### Backtesting

```python
from quantfin.portfolio.backtest import StrategyBacktest, BenchmarkComparison

# Create backtest
backtest = StrategyBacktest(returns_df, rebalance_freq='M', transaction_cost=0.001)

# Run strategies
result_ew = backtest.equal_weight()
result_iv = backtest.inverse_volatility(window=63)
result_rp = backtest.risk_parity(window=63)
result_mv = backtest.minimum_variance(window=63)

# Compare with benchmark
comparison = BenchmarkComparison(result_rp, benchmark_returns)
table = comparison.comparison_table()
rolling = comparison.rolling_performance(window=63)
```

---

## Strategies Module

### Topology Alpha Strategy

```python
from quantfin.strategies.topology import TopologyAlphaStrategy

# Create strategy
strategy = TopologyAlphaStrategy(
    stocks=['AAPL', 'MSFT', 'GOOGL', ...],
    window_size=90,
    vix_threshold=35,
    betti_threshold_low=30,
    betti_threshold_high=45,
    diffusion_time=0.5
)

# Run backtest
results = strategy.backtest(
    prices=prices_df,
    vix=vix_series,
    transaction_cost=0.0005,
    initial_capital=100000.0
)

# Get signals
signals = strategy.generate_signals(returns_df, vix_series)

# Get current weights
weights = strategy.get_weights(returns_window, vix_current=20)
```

---

## Backtesting Module

### Event-Driven Engine

```python
from quantfin.backtesting.engine import BacktestEngine, Order, OrderSide

# Create engine
engine = BacktestEngine(
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0005
)

# Define signal generator
def signal_generator(timestamp, prices_history):
    # Your strategy logic here
    return {'AAPL': 100, 'MSFT': 50}  # Target positions

# Run backtest
equity_curve = engine.run(prices_df, signal_generator, rebalance_freq='M')
metrics = engine.get_performance_metrics()
```

### Performance Metrics

```python
from quantfin.backtesting.metrics import PerformanceMetrics

# Calculate metrics
metrics = PerformanceMetrics(
    returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.045
)

# Get summary
summary = metrics.summary()
monthly = metrics.monthly_returns()
rolling = metrics.rolling_metrics(window=63)
```

---

## Examples

### Example 1: Options Pricing

See `examples/01_options_pricing.py` for complete demonstration of:
- Black-Scholes pricing
- Greeks calculation
- Model comparison (Binomial, Monte Carlo)
- Implied volatility
- Options strategies

### Example 2: Portfolio Optimization

See `examples/02_portfolio_optimization.py` for:
- Mean-variance optimization
- Risk parity
- CVaR optimization
- Efficient frontier
- Strategy backtesting

### Example 3: Topology Alpha Strategy

See `examples/03_topology_alpha.py` for:
- TDA-based signal generation
- Regime detection
- Risk-parity weighting
- Performance analysis

---

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `sharpe_ratio(returns, rf)` | Calculate Sharpe ratio |
| `maximum_drawdown(equity)` | Calculate max drawdown |
| `var_historical(returns, conf)` | Historical VaR |
| `cvar_historical(returns, conf)` | Historical CVaR |
| `beta(returns, benchmark)` | Calculate beta |
| `alpha(returns, benchmark, rf)` | Calculate Jensen's alpha |

### Options Models

| Class | Description |
|-------|-------------|
| `BlackScholes(S, K, T, r, sigma)` | Black-Scholes model |
| `BinomialModel(S, K, T, r, sigma, n)` | CRR binomial model |
| `TrinomialModel(S, K, T, r, sigma, n)` | Trinomial model |
| `MonteCarloOption(S, K, T, r, sigma)` | Monte Carlo pricing |

### Portfolio Optimization

| Class | Description |
|-------|-------------|
| `PortfolioOptimizer(returns)` | Portfolio optimizer |
| `RiskMetrics(returns)` | Risk calculations |
| `RiskParity(returns)` | Risk parity weights |
| `PerformanceAttribution(...)` | Return attribution |

### Strategies

| Class | Description |
|-------|-------------|
| `TopologyAlphaStrategy(...)` | TDA-based strategy |

---

## Advanced Topics

### Custom Strategies

```python
from quantfin.strategies.base import Strategy

class MyStrategy(Strategy):
    def fit(self, data):
        # Training logic
        pass
    
    def generate_signals(self, data):
        # Signal generation
        return [Signal(...)]
    
    def get_weights(self, data):
        # Return portfolio weights
        return pd.Series({...})
```

### Custom Options Models

```python
class MyOptionModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def price(self, option_type='call'):
        # Your pricing formula
        return price
```

---

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Data fetching**: Check API keys and internet connection
3. **Optimization failures**: Try different initial guesses or constraints
4. **Memory issues**: Reduce Monte Carlo simulations or data size

### Performance Tips

1. Use caching for data fetching
2. Vectorize calculations with NumPy
3. Use numba for JIT compilation
4. Parallelize independent calculations

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## License

MIT License - See LICENSE file for details