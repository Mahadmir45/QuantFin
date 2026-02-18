import numpy as np
from scipy.stats import norm
from scipy.optimize import newton, brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

plt.style.use('ggplot')

# ────────────────────────────────────────────────
# Parameters (updated with real data as of Feb 18, 2026)
# ────────────────────────────────────────────────
S_current = 688.6          # Current SPY price
r = 0.042                  # approximate short-term rate ~4.2%
sigma_input = 0.15         # rough ATM IV level for SPY (actual ATM IV ~16%)

K_range = np.linspace(S_current * 0.8, S_current * 1.2, 41)  # reasonable range
T_range = np.array([7/365, 30/365, 90/365, 180/365, 365/365])  # short to 1 year

# ────────────────────────────────────────────────
# Black-Scholes European
# ────────────────────────────────────────────────
def black_scholes(S, K, T, r, sigma, type_="c"):
    if T <= 0:
        return max(S - K, 0) if type_ == "c" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type_ == "c":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ────────────────────────────────────────────────
# Implied Volatility solver
# ────────────────────────────────────────────────
def implied_vol(S, K, T, r, market_price, type_="c", sigma_guess=0.2):
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, type_) - market_price
    try:
        return brentq(objective, 1e-6, 5.0)
    except:
        return np.nan

# ────────────────────────────────────────────────
# Binomial (CRR) - now with American early exercise
# ────────────────────────────────────────────────
def binomial_price(S, K, T, r, sigma, n, type_="c", american=False):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Stock prices at maturity
    stock = S * d**np.arange(n, -1, -1) * u**np.arange(0, n+1)

    # Option values at maturity
    if type_ == "c":
        option = np.maximum(stock - K, 0)
    else:
        option = np.maximum(K - stock, 0)

    # Backward induction
    for i in range(n-1, -1, -1):
        option = discount * (p * option[1:] + (1-p) * option[:-1])
        stock = stock[:-1] / d   # or * u, same recombining
        if american:
            intrinsic = np.maximum(stock - K, 0) if type_ == "c" else np.maximum(K - stock, 0)
            option = np.maximum(option, intrinsic)

    return option[0]

# ────────────────────────────────────────────────
# Trinomial - with American early exercise
# ────────────────────────────────────────────────
def trinomial_price(S, K, T, r, sigma, n, type_="c", american=False):
    dt = T / n
    dx = sigma * np.sqrt(3 * dt)
    nu = (r - 0.5 * sigma**2) * dt
    pu = 0.5 * ((sigma**2 * dt + nu**2) / dx**2 + nu / dx)
    pd = 0.5 * ((sigma**2 * dt + nu**2) / dx**2 - nu / dx)
    pm = 1 - (pu + pd)
    discount = np.exp(-r * dt)

    num_nodes = 2 * n + 1
    center = n

    # Terminal stock prices
    stock = S * np.exp(np.arange(-n, n+1) * dx)

    # Terminal option values
    if type_ == "c":
        v = np.maximum(stock - K, 0)
    else:
        v = np.maximum(K - stock, 0)

    # Backward
    for step in range(n):
        v_new = np.zeros(num_nodes)
        for k in range(1, num_nodes-1):
            cont = pu * v[k+1] + pm * v[k] + pd * v[k-1]
            v_new[k] = discount * cont
        # boundaries (simplified)
        v_new[0] = discount * (pu * v[1] + pm * v[0] + pd * v[0])
        v_new[-1] = discount * (pu * v[-1] + pm * v[-1] + pd * v[-2])
        v = v_new

        # Early exercise
        if american:
            current_stock = S * np.exp((np.arange(-n+step, n-step+1)) * dx)
            intrinsic = np.maximum(current_stock - K, 0) if type_ == "c" else np.maximum(K - current_stock, 0)
            v = np.maximum(v, intrinsic)

    return v[center]

# ────────────────────────────────────────────────
# Speed comparison
# ────────────────────────────────────────────────
n_strikes = 10000
n_steps = 100

print("Timing comparison (n_strikes = 10,000, n_steps=100):")
t0 = time.time()
_ = [black_scholes(S_current, k, T_range[2], r, sigma_input) for k in np.linspace(S_current*0.5, S_current*1.5, n_strikes)]
print(f"Black-Scholes: {time.time()-t0:.4f} s")

t0 = time.time()
_ = [binomial_price(S_current, k, T_range[2], r, sigma_input, n_steps, "c") for k in np.linspace(S_current*0.5, S_current*1.5, n_strikes)]
print(f"Binomial European: {time.time()-t0:.4f} s")

t0 = time.time()
_ = [trinomial_price(S_current, k, T_range[2], r, sigma_input, n_steps, "c") for k in np.linspace(S_current*0.5, S_current*1.5, n_strikes)]
print(f"Trinomial European: {time.time()-t0:.4f} s")

# American is slower due to max() at each node
t0 = time.time()
_ = [binomial_price(S_current, k, T_range[2], r, sigma_input, n_steps, "c", american=True) for k in np.linspace(S_current*0.5, S_current*1.5, n_strikes//10)]
print(f"Binomial American (10% sample): {time.time()-t0:.4f} s")

# ────────────────────────────────────────────────
# Implied Volatility Surface (using BS as "market" for demo)
# In real use → replace black_scholes(...) with actual market mid-price
# ────────────────────────────────────────────────
IV_surface = np.zeros((len(T_range), len(K_range)))

for i, T in enumerate(T_range):
    for j, K in enumerate(K_range):
        model_price = black_scholes(S_current, K, T, r, sigma_input)  # proxy for market
        IV_surface[i, j] = implied_vol(S_current, K, T, r, model_price)

# 2D Smile (for one maturity)
plt.figure(figsize=(10,6))
for i, T in enumerate(T_range[::2]):  # every other for clarity
    plt.plot(K_range / S_current, IV_surface[i], label=f"T = {T*365:.0f} days")
plt.axvline(1.0, color='gray', ls='--', label="ATM")
plt.title("Implied Volatility Smile / Skew (BS flat vol)")
plt.xlabel("Moneyness (K/S)")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()

# 3D Surface
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
K_mesh, T_mesh = np.meshgrid(K_range, T_range*365)
ax.plot_surface(K_mesh, T_mesh, IV_surface*100, cmap='viridis')
ax.set_xlabel("Strike")
ax.set_ylabel("Days to Expiration")
ax.set_zlabel("Implied Vol (%)")
ax.set_title("Implied Volatility Surface (flat input vol → flat surface)")
plt.show()

# ────────────────────────────────────────────────
# Market data comparison (real data for March 20, 2026 expiration ~30 days)
# ────────────────────────────────────────────────
real_market = {
    "T": 30/365,
    "K": [650, 680, 690, 700, 720],
    "call_mid": [44.055, 19.185, 12.195, 6.645, 0.985],
    "put_mid": [3.925, 9.35, 12.595, 17.265, 32.195]
}

model_prices_call = [black_scholes(S_current, k, real_market["T"], r, sigma_input) for k in real_market["K"]]
errors_call = np.array(real_market["call_mid"]) - np.array(model_prices_call)

print("\nReal market vs BS (March 20, 2026 expiration):")
for k, mkt, mod, err in zip(real_market["K"], real_market["call_mid"], model_prices_call, errors_call):
    print(f"K={k:>5}  Market={mkt:5.2f}  Model={mod:5.2f}  Error={err:6.2f}")

mae = np.mean(np.abs(errors_call))
rmse = np.sqrt(np.mean(errors_call**2))
print(f"MAE: {mae:.3f}   RMSE: {rmse:.3f}")

# IV RMSE would be: compute IV from market prices, compare to model IVs

# ────────────────────────────────────────────────
# Heston stochastic volatility (basic - for smile/skew)
# ────────────────────────────────────────────────
# Very simplified Heston call price (semi-closed form via Fourier or numerical integration)
# For full production use: use libraries like QuantLib or implement Carr-Madan FFT
# Here: just a placeholder showing you can get skew/smile

def heston_call_approx(S, K, T, r, v0, kappa, theta, sigma_v, rho):
    # Very rough approximation / placeholder
    # Real implementation needs numerical integration or FFT
    # For illustration: increase vol-of-vol & negative rho → skew
    avg_vol = np.sqrt((v0 + (theta-v0)*(1-np.exp(-kappa*T))/ (kappa*T) ))
    price_bs = black_scholes(S, K, T, r, avg_vol)
    skew_adjust = rho * 0.1 * (K/S - 1) * np.sqrt(T)   # toy skew
    return price_bs * (1 + skew_adjust)

# Example: negative rho creates put skew (higher IV for low strikes)
print("\nHeston toy example (negative rho → put skew):")
for K in [S_current*0.9, S_current, S_current*1.1]:
    iv_heston = implied_vol(S_current, K, 0.25, r,
                            heston_call_approx(S_current, K, 0.25, r, 0.04, 2.0, 0.04, 0.5, -0.7),
                            "c")
    print(f"K = {K:.0f} → IV ≈ {iv_heston:.1%}")

# Local volatility (Dupire) stub:
# σ_local(K,T) = sqrt( ∂C/∂T / (0.5 K² ∂²C/∂K²) )   # requires market surface of calls
# Implementation needs smooth implied vol surface + numerical derivatives
print("\nLocal vol / Dupire: requires full market IV surface + smoothing + differentiation.")
print("Typical use: calibrate dupire local vol to reproduce smile/skew observed in market.")
