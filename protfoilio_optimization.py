import logging
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import yfinance as yf
import shutil
import os

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Starting Quant Finance Project")

# Clear yfinance cache
cache_dir = os.path.expanduser("~/.cache/yfinance")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)
    logging.info("Cleared yfinance cache")

# Parameters
S0 = 26700.0
K = 26700.0
T = 1.0
r = 0.035
sigma_input = 0.205


# Robust Data Fetch
def fetch_returns(tickers, period="2y"):
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, threads=False)['Close']
        returns = data.pct_change().dropna()
        if returns.empty or returns.isna().any().any():
            raise ValueError
        logging.info("Successfully fetched real market data from yfinance")
        return returns
    except Exception as e:
        logging.warning(f"yfinance failed: {e}. Using synthetic HK data")
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=504, freq='B')
        hsi_ret = np.random.normal(0.0004, 0.0125, 504)
        tencent_ret = np.random.normal(0.0006, 0.018, 504)
        alibaba_ret = np.random.normal(0.0003, 0.022, 504)
        returns = pd.DataFrame({
            '^HSI': hsi_ret,
            '0700.HK': tencent_ret,
            '9988.HK': alibaba_ret
        }, index=dates)
        return returns


tickers = ['^HSI', '0700.HK', '9988.HK']
returns_df = fetch_returns(tickers)

mu = returns_df.mean() * 252
cov = returns_df.cov() * 252

try:
    S0 = yf.Ticker('^HSI').history(period="5d")['Close'].iloc[-1]
except:
    pass

sigma = returns_df['^HSI'].std() * np.sqrt(252)
if np.isnan(sigma) or sigma <= 0:
    sigma = sigma_input

logging.info(f"HSI price = {S0:.1f} | Annualized volatility = {sigma:.1%}")


# Monte Carlo with optional path return
def monte_carlo_option(S0, K, T, r, sigma, num_sims=200000, steps=252,
                       option_type='european', payoff_type='call', return_paths=False):
    dt = T / steps
    z = np.random.normal(0, 1, (num_sims, steps))
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.column_stack((np.full(num_sims, S0), paths))

    if option_type == 'asian':
        avg = np.mean(paths, axis=1)
        payoff = np.maximum(avg - K, 0) if payoff_type == 'call' else np.maximum(K - avg, 0)
    elif option_type == 'lookback':
        extremum = np.max(paths, axis=1) if payoff_type == 'call' else np.min(paths, axis=1)
        payoff = np.maximum(extremum - K, 0) if payoff_type == 'call' else np.maximum(K - extremum, 0)
    else:
        final = paths[:, -1]
        payoff = np.maximum(final - K, 0) if payoff_type == 'call' else np.maximum(K - final, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    std_err = np.std(payoff) / np.sqrt(num_sims)

    if return_paths:
        return price, std_err, paths
    return price, std_err


# Robust Portfolio Optimizer
def make_positive_definite(cov_matrix, epsilon=1e-5):
    return cov_matrix + epsilon * np.eye(cov_matrix.shape[0])


def optimize_portfolio(mu, cov, target_return=0.12, max_w=0.35, use_cvar=True):
    cov = make_positive_definite(cov)
    n = len(mu)

    def objective(w):
        if use_cvar:
            try:
                sim_r = np.random.multivariate_normal(mu, cov, 12000)
                port = sim_r @ w
                var = np.percentile(port, 5)
                return -np.mean(port[port <= var])
            except:
                logging.warning("CVaR failed, falling back to variance minimization")
                return w.T @ cov @ w
        return w.T @ cov @ w

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if target_return is not None:
        cons.append({'type': 'eq', 'fun': lambda w: w @ mu - target_return})

    bounds = [(0, max_w)] * n
    res = minimize(objective, np.ones(n) / n, bounds=bounds, constraints=cons, tol=1e-9)

    if res.success:
        w = res.x
        ret = w @ mu
        risk = -res.fun if use_cvar else np.sqrt(w.T @ cov @ w)
        logging.info("Portfolio optimization successful")
        return w, ret, risk
    else:
        logging.warning("Optimizer failed - using equal weight fallback")
        w = np.ones(n) / n
        return w, w @ mu, np.sqrt(w.T @ cov @ w)


# Run Monte Carlo
logging.info("Running Monte Carlo pricing...")
mc_euro, err = monte_carlo_option(S0, K, T, r, sigma)
bs_price = S0 * norm.cdf((np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) - \
           K * np.exp(-r * T) * norm.cdf((np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))

mc_asian, _ = monte_carlo_option(S0, K, T, r, sigma, option_type='asian')
mc_lookback, _ = monte_carlo_option(S0, K, T, r, sigma, option_type='lookback')

logging.info(f"European Call (MC)  : {mc_euro:8.2f} ± {err:.4f}")
logging.info(f"European Call (BS)  : {bs_price:8.2f} ← Benchmark")
logging.info(f"Asian Call          : {mc_asian:8.2f}")
logging.info(f"Lookback Call       : {mc_lookback:8.2f}")

# Portfolio
logging.info("Optimizing portfolio...")
w, ret, risk = optimize_portfolio(mu, cov, target_return=0.12)

logging.info("Optimal HK Portfolio")
for t, weight in zip(tickers, w):
    logging.info(f"{t:12} : {weight:7.2%}")
logging.info(f"Expected Return : {ret:.2%}")
logging.info(f"Risk (CVaR 5%)  : {risk:.2%}")

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Portfolio pie
axs[0].pie(w, labels=tickers, autopct='%1.1f%%')
axs[0].set_title('Optimal HK Portfolio')

# Simulated paths - correct way
_, _, paths = monte_carlo_option(S0, K, T, r, sigma, num_sims=8, return_paths=True)
paths_plot = paths[:, :252].T
axs[1].plot(paths_plot, alpha=0.7)
axs[1].set_title('Simulated HSI Paths (1 Year)')
axs[1].set_xlabel('Trading Days')
axs[1].set_ylabel('HSI Level')

plt.tight_layout()
plt.show()

logging.info("Project completed successfully")