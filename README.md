Topology Alpha Model

A quantitative trading strategy using **Topological Data Analysis (TDA)** on stock correlation graphs to detect market regimes, generate alpha signals via Laplacian diffusion residuals, and time exposure with VIX filtering. The model employs risk-parity (inverse volatility) weighting for diversified long-only positions in low-risk regimes.

The goal is consistent outperformance of SPY with superior risk-adjusted returns (higher Sharpe, lower drawdown).

Features

- **Algebraic Topology Core**:
  - Correlation graph construction.
  - Persistent homology proxy (Betti numbers & lifetimes).
  - Laplacian diffusion for residual alpha (local mispricing).
- **Regime Timing**:
  - Low std_betti + low VIX = aggressive exposure.
  - High risk = flat (protects in volatility).
- **Positioning**:
  - Inverse volatility (risk parity) weights.
  - Adaptive exposure (1.2 in strong low-risk, 0.6 minimum).
- **Backtesting**:
  - 25 diversified US stocks + SPY benchmark + VIX.
  - Transaction costs included.
  - Equity curve plotting.

## Installation


python -m venv .venv
source .venv/bin/activate
pip install numpy pandas networkx scipy massive matplotlib


Your Polygon (Massive) API key is required in the script (`API_KEY` variable).

## Usage


python script.py --backtest
# Optional: --force-refresh to update data cache


The script fetches historical data (cached for speed), runs the backtest, and plots the equity curve vs SPY.

## Current Performance (example from backtest)

- **Strategy**: Sharpe ~0.6, Max DD ~8-12%, PNL ~4-8% (period-dependent)
- **SPY B&H**: Sharpe ~0.2, Max DD ~22%, PNL ~4%

The model excels in risk control while capturing upside in stable regimes.

## Repository Structure

- `script.py`: Main backtest code.
- `historical_prices.csv`: Cached price data (auto-generated).
- `README.md`: This file.

## Upcoming Enhancements

To make this a stronger, production-ready quant model:

1. **Realtime Trading Integration**:
   - Alpaca API for live execution.
   - Daily run with current data and order submission in low-risk regimes.

2. **Sentiment Analysis**:
   - X (Twitter) semantic search for real-time market sentiment.
   - Boost exposure on positive sentiment, reduce on negative.

3. **Full Persistence Diagrams**:
   - Use ripser/gudhi for complete homology computation (H0/H1 persistence landscapes as features).

4. **Advanced ML**:
   - LSTM or Transformer on time-series of topology features + residuals for better alpha prediction.
   - Ensemble with current rule-based timing.

5. **Dynamic Optimization**:
   - Adaptive thresholds (rolling mean/std of std_betti/VIX).
   - Mean-variance optimization in bull regimes.

6. **Risk Management**:
   - Trailing stop-loss.
   - Position sizing based on predicted volatility.

7. **Extended Universe**:
   - More stocks/ETFs for better diversification.
   - Sector rotation using topology clusters.

8. **Metrics & Logging**:
   - Full performance report (Calmar, Sortino, turnover).
   - Logging to file for paper trading.

Contributions welcome! This is an evolving research-grade quant model using cutting-edge topological methods in finance.

