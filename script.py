import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import expm, eigh
from massive import RESTClient
import time
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor


API_KEY = 'bhRlfQ4mbGkCYvG5NG9nsV0ABMaKnZWT'


CACHE_FILE = 'historical_prices.csv'


stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'BAC', 'GS',
          'JNJ', 'PFE', 'LLY', 'XOM', 'CVX', 'PG', 'KO', 'PEP', 'MCD', 'SBUX',
          'BA', 'CAT', 'DUK', 'DD', 'WMT']
all_tickers = stocks + ['SPY', 'VIX']



def get_historical_prices(from_date='2022-01-01', to_date='today', force_refresh=False):
    if to_date == 'today':
        to_date = datetime.now().strftime('%Y-%m-%d')

    prices = pd.DataFrame()

    if os.path.exists(CACHE_FILE) and not force_refresh:
        print(f"Loading cached prices from {CACHE_FILE}...")
        prices = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        prices.index = pd.to_datetime(prices.index)
        last_date = prices.index[-1].strftime('%Y-%m-%d')
        print(f"Last cached date: {last_date}")

        if last_date >= to_date:
            print("Cache is up to date.")
            return prices

        fetch_from = (prices.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Attempting incremental fetch from {fetch_from} to {to_date}...")
    else:
        print("Force refresh or no cache: fetching full history...")
        fetch_from = from_date

    client = RESTClient(API_KEY)
    new_data = {}
    success = False
    for ticker in all_tickers:
        aggs = []
        try:
            for a in client.list_aggs(ticker, 1, 'day', fetch_from, to_date, limit=50000):
                aggs.append(a)
            if aggs:
                success = True
            print(f"Fetched {ticker} ({len(aggs)} new bars)")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue
        if aggs:
            df = pd.DataFrame(aggs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            new_data[ticker] = df['close']
        time.sleep(13)

    if new_data and success:
        new_prices = pd.DataFrame(new_data)
        prices = pd.concat([prices, new_prices]).sort_index()
        prices = prices[~prices.index.duplicated(keep='last')]
        print(f"Updated with new data ({len(new_prices)} days added).")
    else:
        print("No new data fetched (free plan limit or current day unavailable). Using cached prices up to last date.")

    print(f"Saving cache to {CACHE_FILE} ({len(prices)} days total)...")
    prices.to_csv(CACHE_FILE)

    return prices



def prices_to_returns(prices):
    returns = prices.pct_change().dropna()
    return returns



def build_graph(returns_window):
    corr = returns_window.corr().abs().fillna(0)
    A = corr.values
    D = np.diag(A.sum(axis=1))
    L = D - A
    return L, corr.values



def laplacian_diffusion(L, signal, t=0.5):
    diffused = expm(-t * L) @ signal
    return diffused



def compute_ph_features(corr_matrices):
    features = []
    thresholds = np.linspace(0.1, 0.9, 7)
    for corr in corr_matrices:
        ph_vec = []
        for thresh in thresholds:
            adj = (corr > thresh).astype(int) - np.eye(len(corr))
            G = nx.from_numpy_array(adj)
            betti0 = nx.number_connected_components(G) - 1
            n_nodes = len(corr)
            n_edges = G.number_of_edges()
            betti1 = max(0, n_edges - n_nodes + betti0 + 1)
            ph_vec.extend([betti0, betti1])
        lifetimes = np.abs(np.diff(ph_vec + [0]))
        features.append(np.hstack([
            np.mean(ph_vec), np.std(ph_vec), np.mean(lifetimes), np.max(lifetimes),
            np.sum(lifetimes), np.median(lifetimes)
        ]))
    return np.array(features)



def backtest(prices, window_size=90, tx_cost=0.0005, retrain_every=15):
    returns = prices_to_returns(prices)
    stock_returns = returns[stocks]
    spy_returns = returns['SPY']
    vix = prices.get('VIX', pd.Series(20 * np.ones(len(prices)), index=prices.index))
    equity_strategy = [1.0]
    equity_spy = [1.0]
    dates = stock_returns.index[window_size + 1:]
    model = None
    print("Running risk parity + high-activity topology model backtest...")
    for i in range(window_size, len(stock_returns) - 1):
        past_returns = stock_returns.iloc[i - window_size:i]
        L, corr = build_graph(past_returns)
        current_returns = stock_returns.iloc[i].values
        diffused = laplacian_diffusion(L, current_returns, t=0.5)
        residuals = current_returns - diffused
        eigenvalues = eigh(L)[0][:3]
        corr_matrices = [corr]
        ph_features = compute_ph_features(corr_matrices)[0]
        std_betti = ph_features[1]
        vix_level = vix.iloc[i]
        mom = past_returns.mean().mean()
        global_features = np.hstack([eigenvalues, ph_features, vix_level, mom])

        if model is None or (i - window_size) % retrain_every == 0:
            X_hist = []
            y_hist = []
            for j in range(window_size + 30, i):
                past_j = stock_returns.iloc[j - window_size:j]
                L_j, _ = build_graph(past_j)
                curr_j = stock_returns.iloc[j].values
                diff_j = laplacian_diffusion(L_j, curr_j, t=0.5)
                res_j = curr_j - diff_j
                eig_j = eigh(L_j)[0][:3]
                ph_j = compute_ph_features([past_j.corr().values])[0]
                vix_j = vix.iloc[j]
                mom_j = past_j.mean().mean()
                X_hist.append(np.hstack([res_j, eig_j, ph_j, vix_j, mom_j]))
                y_hist.append(stock_returns.iloc[j + 1].values)
            if len(X_hist) > 100:
                model = XGBRegressor(n_estimators=150, objective='reg:squarederror', n_jobs=-1)
                model.fit(X_hist, y_hist)
                print(f"Retrained at day {i}")

        vol = past_returns.std().values + 1e-6
        risk_parity_weights = 1 / vol
        risk_parity_weights /= risk_parity_weights.sum()

        if vix_level > 35 and std_betti > 45:
            daily_ret = 0.0
        else:
            if model is not None:
                features = np.hstack([residuals, global_features]).reshape(1, -1)
                pred_alpha = model.predict(features)[0]
            else:
                pred_alpha = residuals

            sorted_idx = np.argsort(pred_alpha)[-25:]
            weights = np.ones(25) / 25
            exposure = 1.2 if std_betti < 30 else 1.0 if std_betti < 40 else 0.6
            daily_ret = np.sum(weights * stock_returns.iloc[i + 1].values[sorted_idx]) * exposure - tx_cost
        equity_strategy.append(equity_strategy[-1] * (1 + daily_ret))
        equity_spy.append(equity_spy[-1] * (1 + spy_returns.iloc[i + 1] - tx_cost))
    equity_strategy = np.array(equity_strategy[1:])
    equity_spy = np.array(equity_spy[1:])
    total_pnl_strategy = equity_strategy[-1] - 1
    total_pnl_spy = equity_spy[-1] - 1
    daily_rets_strategy = np.diff(equity_strategy) / equity_strategy[:-1]
    daily_rets_spy = np.diff(equity_spy) / equity_spy[:-1]
    sharpe_strategy = np.mean(daily_rets_strategy) / np.std(daily_rets_strategy) * np.sqrt(252) if np.std(
        daily_rets_strategy) > 0 else 0
    sharpe_spy = np.mean(daily_rets_spy) / np.std(daily_rets_spy) * np.sqrt(252) if np.std(daily_rets_spy) > 0 else 0
    dd_strategy = np.min(
        np.cumprod(1 + daily_rets_strategy) / np.maximum.accumulate(np.cumprod(1 + daily_rets_strategy)) - 1)
    dd_spy = np.min(np.cumprod(1 + daily_rets_spy) / np.maximum.accumulate(np.cumprod(1 + daily_rets_spy)) - 1)
    print(f"Strategy: Sharpe {sharpe_strategy:.2f}, DD {dd_strategy:.2%}, PNL {(total_pnl_strategy * 100):.2f}%")
    print(f"SPY B&H: Sharpe {sharpe_spy:.2f}, DD {dd_spy:.2%}, PNL {(total_pnl_spy * 100):.2f}%")
    pd.Series(equity_strategy, index=dates).plot(label='Risk Parity Topology Model')
    pd.Series(equity_spy, index=dates).plot(label='SPY Buy & Hold')
    plt.legend()
    plt.title('Equity Curve: Strategy vs SPY')
    plt.ylabel('Equity (starting 1.0)')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--force-refresh', action='store_true')
    args = parser.parse_args()

    if args.backtest:
        prices = get_historical_prices(force_refresh=args.force_refresh)
        backtest(prices)
    else:
        print("Run with --backtest [--force-refresh]")