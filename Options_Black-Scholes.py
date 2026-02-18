
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Model Parameters
r = 0.055  # Risk-free interest rate
S = 600  # Current stock price
K = 600  # Strike price
T = 365 / 365  # Time to expiration (in years)
sigma = 0.1625  # Volatility

K_range = np.linspace(400, 800, 500)  # Strike prices from 400 to 800


# Black-Scholes Option Pricing Function
def blackScholes(r, S, K, T, sigma, type="c"):
    """
    Calculate the theoretical price of a call or put option using Black-Scholes model

    Parameters:
    - r: Risk-free interest rate
    - S: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - sigma: Volatility of the underlying asset
    - type: Option type - 'c' for Call, 'p' for Put

    Returns:
    - Theoretical option price
    """
    # Calculate d1 and d2 parameters
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            # Call option pricing formula
            price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            # Put option pricing formula
            price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
        return price
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")


# Calculate call and put option prices
call_prices = [blackScholes(r, S, K, T, sigma, type="c") for K in K_range]
put_prices = [blackScholes(r, S, K, T, sigma, type="p") for K in K_range]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, call_prices, label="Call Option Price", color="blue")
plt.plot(K_range, put_prices, label="Put Option Price", color="red")
plt.axvline(x=K, color="gray", linestyle="--", label="Strike Price (K)")
plt.title("Black-Scholes Option Pricing")
plt.xlabel("Strike Price (K)")
plt.ylabel("Option Price")
plt.legend()
plt.show()


# Option Greeks Calculation Functions
# Delta
# Delta measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price.

def delta_calc(r, S, K, T, sigma, type="c"):
    """
    Calculate Delta: Rate of change of option price with respect to underlying asset price

    Delta represents the hedge ratio or the equivalent stock position of the option
    - For calls: Ranges from 0 to 1
    - For puts: Ranges from -1 to 0

    Parameters same as Black-Scholes function

    Returns:
    - Delta value
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    try:
        if type == "c":
            delta_calc = norm.cdf(d1, 0, 1)
        elif type == "p":
            delta_calc = -norm.cdf(-d1, 0, 1)
        return delta_calc
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")


# Calculate Delta values for calls and puts
call_deltas = [delta_calc(r, S, K, T, sigma, type="c") for K in K_range]
put_deltas = [delta_calc(r, S, K, T, sigma, type="p") for K in K_range]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, call_deltas, label="Delta (Call)", color="blue")
plt.plot(K_range, put_deltas, label="Delta (Put)", color="red")
plt.axvline(x=K, color="gray", linestyle="--", label="Strike Price (K)")
plt.title("Delta vs Strike Price (K)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Delta")
plt.legend()
plt.show()


# Gamma
# Gamma measures the rate of change in the delta with respect to changes in the underlying price.

def gamma_calc(r, S, K, T, sigma, type="c"):
    """
    Calculate Gamma: Rate of change of Delta with respect to underlying asset price

    Gamma measures the curvature of the option price's relationship to the underlying price
    - Highest near the money
    - Symmetric for calls and puts

    Parameters same as Black-Scholes function

    Returns:
    - Gamma value
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    try:
        gamma_calc = norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(T))
        return gamma_calc
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")


# Calculate Gamma values for the stock price range
gamma_values = [gamma_calc(r, S, K, T, sigma) for K in K_range]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, gamma_values, label="Gamma (Call & Put)", color="purple")
plt.axvline(x=K, color="gray", linestyle="--", label="Strike Price (K)")
plt.title("Gamma vs Strike Price (K)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Gamma")
plt.legend()
plt.show()


# Vega
# Vega measures sensitivity to volatility. Vega is the derivative of the option value with respect to the volatility of the underlying asset.

def vega_calc(r, S, K, T, sigma, type="c"):
    """
    Calculate Vega: Sensitivity of option price to volatility changes

    Vega measures how much an option's price changes with volatility
    - Highest for at-the-money options
    - Multiplied by 0.01 to represent percentage point change

    Parameters same as Black-Scholes function

    Returns:
    - Vega value
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    try:
        vega_calc = S * norm.pdf(d1, 0, 1) * np.sqrt(T)
        return vega_calc * 0.01  # Convert to percentage points
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")


# Calculate Vega values for the stock price range
vega_values = [vega_calc(r, S, K, T, sigma) for K in K_range]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, vega_values, label="Vega (Call & Put)", color="green")
plt.axvline(x=K, color="gray", linestyle="--", label="Strike Price (K)")
plt.title("Vega vs Strike Price (K)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Vega")
plt.legend()
plt.show()


# Theta
# Theta measures the sensitivity of the value of the derivative to the passage of time - time decay.

def theta_calc(r, S, K, T, sigma, type="c"):
    """
    Calculate Theta: Rate of time decay of the option

    Theta measures how much value an option loses as time passes
    - Typically negative (option loses value as expiration approaches)
    - Divided by 365 to get daily time decay

    Parameters same as Black-Scholes function

    Returns:
    - Theta value (daily time decay)
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    try:
        if type == "c":
            theta_calc = -S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2, 0,
                                                                                                                1)
        elif type == "p":
            theta_calc = -S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2, 0,
                                                                                                                1)
        return theta_calc / 365  # Daily time decay
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")


# Calculate Theta values for the strike price range
theta_values_call = [theta_calc(r, S, K, T, sigma, type="c") for K in K_range]
theta_values_put = [theta_calc(r, S, K, T, sigma, type="p") for K in K_range]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, theta_values_call, label="Theta (Call)", color="blue")
plt.plot(K_range, theta_values_put, label="Theta (Put)", color="red")
plt.axvline(x=S, color="gray", linestyle="--", label="Stock Price (S)")
plt.title("Theta vs Strike Price (K)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Theta")
plt.legend()
plt.show()


# Rho
# Rho measures the sensitivity to the interest rate.

def rho_calc(r, S, K, T, sigma, type="c"):
    """
    Calculate Rho: Sensitivity of option price to interest rate changes

    Rho measures how much an option's price changes with interest rates
    - Multiplied by 0.01 to represent percentage point change

    Parameters same as Black-Scholes function

    Returns:
    - Rho value
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    try:
        if type == "c":
            rho_calc = K * T * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            rho_calc = -K * T * np.exp(-r * T) * norm.cdf(-d2, 0, 1)
        return rho_calc * 0.01  # Convert to percentage points
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")


# Calculate Rho values for the strike price range
rho_values_call = [rho_calc(r, S, K, T, sigma, type="c") for K in K_range]
rho_values_put = [rho_calc(r, S, K, T, sigma, type="p") for K in K_range]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(K_range, rho_values_call, label="Rho (Call)", color="blue")
plt.plot(K_range, rho_values_put, label="Rho (Put)", color="red")
plt.axvline(x=S, color="gray", linestyle="--", label="Stock Price (S)")
plt.title("Rho vs Strike Price (K)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Rho")
plt.legend()
plt.show()


# Implied Volatility Calculation
# Implied volatility is the volatility that makes the Black-Scholes price equal to the market price.

def implied_vol(r, S, K, T, market_price, type="c", initial_guess=0.2):
    """
    Calculate implied volatility using Newton-Raphson method.

    Solves for sigma such that Black-Scholes price equals the given market price.

    Parameters:
    - r: Risk-free interest rate
    - S: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - market_price: Observed market price of the option
    - type: Option type - 'c' for Call, 'p' for Put
    - initial_guess: Initial guess for volatility

    Returns:
    - Implied volatility
    """

    def objective(sigma):
        return blackScholes(r, S, K, T, sigma, type) - market_price

    try:
        iv = newton(objective, x0=initial_guess)
        return iv
    except:
        print("Implied volatility calculation failed to converge.")
        return None


# Example usage for implied volatility
# Assume a market price for the at-the-money call
atm_call_bs_price = blackScholes(r, S, K, T, sigma, "c")
example_market_price = atm_call_bs_price + 1  # Slightly perturbed for demonstration
implied_sigma = implied_vol(r, S, K, T, example_market_price, "c")
print(f"Implied Volatility for ATM Call (market price {example_market_price}): {implied_sigma}")


# Binomial Option Pricing Model
# The binomial model is a discrete-time model for pricing options. It approximates the continuous
# lognormal distribution of stock prices using a recombining tree. Here, we implement the Cox-Ross-Rubinstein (CRR) model
# for European call and put options. As the number of steps increases, it converges to the Black-Scholes price.

def binomial_price(r, S, K, T, sigma, n_steps, type="c"):
    """
    Calculate the price of a European call or put option using the binomial (CRR) model.

    The model builds a binomial tree of stock prices and works backwards from expiration to compute the option value.

    Parameters:
    - r: Risk-free interest rate
    - S: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - sigma: Volatility of the underlying asset
    - n_steps: Number of time steps in the binomial tree (higher for better accuracy)
    - type: Option type - 'c' for Call, 'p' for Put

    Returns:
    - Option price
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Build stock price tree
    stock = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            stock[j, i] = S * (u ** (i - j)) * (d ** j)

    # Initialize option values at maturity
    option = np.zeros((n_steps + 1, n_steps + 1))
    if type == "c":
        option[:, n_steps] = np.maximum(stock[:, n_steps] - K, 0)
    elif type == "p":
        option[:, n_steps] = np.maximum(K - stock[:, n_steps], 0)
    else:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")
        return None

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            option[j, i] = np.exp(-r * dt) * (p * option[j, i + 1] + (1 - p) * option[j + 1, i + 1])

    return option[0, 0]


# Trinomial Option Pricing Model
# The trinomial model extends the binomial model by allowing three possible price movements: up, middle, down.
# It provides better convergence for the same number of steps. Implemented for European options, approximating
# the lognormal process with adjusted probabilities to match risk-neutral drift and variance.

def trinomial_price(r, S, K, T, sigma, n_steps, type="c"):
    """
    Calculate the price of a European call or put option using the trinomial model.

    The model uses a recombining tree with three branches per node and probabilities adjusted to match
    the mean and variance of the continuous process.

    Parameters:
    - r: Risk-free interest rate
    - S: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - sigma: Volatility of the underlying asset
    - n_steps: Number of time steps in the trinomial tree (higher for better accuracy)
    - type: Option type - 'c' for Call, 'p' for Put

    Returns:
    - Option price
    """
    dt = T / n_steps
    dx = sigma * np.sqrt(3 * dt)
    nu = r - 0.5 * sigma ** 2

    a = (sigma ** 2 * dt + nu ** 2 * dt ** 2) / dx ** 2
    b = nu * dt / dx
    pu = 0.5 * (a + b)
    pd = 0.5 * (a - b)
    pm = 1.0 - a

    discount = np.exp(-r * dt)
    num_nodes = 2 * n_steps + 1
    v = np.zeros(num_nodes)

    # Set up payoff at maturity
    for k in range(num_nodes):
        stock = S * np.exp((k - n_steps) * dx)
        if type == "c":
            v[k] = max(stock - K, 0)
        elif type == "p":
            v[k] = max(K - stock, 0)
        else:
            print("Please confirm option type, either 'c' for Call or 'p' for Put!")
            return None

    # Backward induction
    for _ in range(n_steps):
        v_new = np.zeros(num_nodes)
        for k in range(1, num_nodes - 1):
            v_new[k] = discount * (pu * v[k + 1] + pm * v[k] + pd * v[k - 1])
        # Boundary approximations
        v_new[0] = discount * (pu * v[1] + pm * v[0] + pd * v[0])
        v_new[num_nodes - 1] = discount * (pu * v[num_nodes - 1] + pm * v[num_nodes - 1] + pd * v[num_nodes - 2])
        v = v_new

    return v[n_steps]


# Calculate binomial and trinomial prices with 100 steps
n_steps = 100
binomial_call_prices = [binomial_price(r, S, K_val, T, sigma, n_steps, "c") for K_val in K_range]
binomial_put_prices = [binomial_price(r, S, K_val, T, sigma, n_steps, "p") for K_val in K_range]
trinomial_call_prices = [trinomial_price(r, S, K_val, T, sigma, n_steps, "c") for K_val in K_range]
trinomial_put_prices = [trinomial_price(r, S, K_val, T, sigma, n_steps, "p") for K_val in K_range]

# Plot binomial and trinomial prices vs Black-Scholes
plt.figure(figsize=(10, 6))
plt.plot(K_range, call_prices, label="BS Call", color="blue")
plt.plot(K_range, put_prices, label="BS Put", color="red")
plt.plot(K_range, binomial_call_prices, label="Binomial Call", color="blue", linestyle="--")
plt.plot(K_range, binomial_put_prices, label="Binomial Put", color="red", linestyle="--")
plt.plot(K_range, trinomial_call_prices, label="Trinomial Call", color="blue", linestyle=":")
plt.plot(K_range, trinomial_put_prices, label="Trinomial Put", color="red", linestyle=":")
plt.axvline(x=K, color="gray", linestyle="--", label="Strike Price (K)")
plt.title("Binomial & Trinomial vs Black-Scholes Option Pricing")
plt.xlabel("Strike Price (K)")
plt.ylabel("Option Price")
plt.legend()
plt.show()

# Example: Convergence of binomial and trinomial to Black-Scholes
steps_range = [10, 50, 100, 200]
convergence_bin_calls = [binomial_price(r, S, K, T, sigma, n, "c") for n in steps_range]
convergence_tri_calls = [trinomial_price(r, S, K, T, sigma, n, "c") for n in steps_range]
convergence_bin_puts = [binomial_price(r, S, K, T, sigma, n, "p") for n in steps_range]
convergence_tri_puts = [trinomial_price(r, S, K, T, sigma, n, "p") for n in steps_range]
bs_call = blackScholes(r, S, K, T, sigma, "c")
bs_put = blackScholes(r, S, K, T, sigma, "p")

print("Binomial and Trinomial Convergence to Black-Scholes (ATM):")
for i, n in enumerate(steps_range):
    print(
        f"Steps {n}: Bin Call {convergence_bin_calls[i]:.4f}, Tri Call {convergence_tri_calls[i]:.4f} (BS: {bs_call:.4f}), Bin Put {convergence_bin_puts[i]:.4f}, Tri Put {convergence_tri_puts[i]:.4f} (BS: {bs_put:.4f})")

# Calculate implied volatilities for different strikes
# Here, we use trinomial call prices as proxy for "market" prices to demonstrate volatility smile (should be nearly flat since models are similar)
implied_vols = []
for i, K_val in enumerate(K_range):
    market_price = trinomial_call_prices[i]
    iv = implied_vol(r, S, K_val, T, market_price, "c", initial_guess=sigma)
    if iv is not None:
        implied_vols.append(iv)
    else:
        implied_vols.append(np.nan)

# Plot implied volatility vs strike
plt.figure(figsize=(10, 6))
plt.plot(K_range, implied_vols, label="Implied Vol (from Trinomial as Market)", color="orange")
plt.axhline(y=sigma, color="gray", linestyle="--", label="Input Volatility")
plt.axvline(x=K, color="gray", linestyle="--", label="Strike Price (K)")
plt.title("Implied Volatility vs Strike Price (K)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()

# Performance metrics
# Compare binomial and trinomial to Black-Scholes as benchmark
bs_call_prices = np.array(call_prices)
bin_call_prices = np.array(binomial_call_prices)
tri_call_prices = np.array(trinomial_call_prices)

mae_bin = np.mean(np.abs(bin_call_prices - bs_call_prices))
rmse_bin = np.sqrt(np.mean((bin_call_prices - bs_call_prices) ** 2))
mae_tri = np.mean(np.abs(tri_call_prices - bs_call_prices))
rmse_tri = np.sqrt(np.mean((tri_call_prices - bs_call_prices) ** 2))

print(f"Binomial vs BS: MAE {mae_bin:.4f}, RMSE {rmse_bin:.4f}")
print(f"Trinomial vs BS: MAE {mae_tri:.4f}, RMSE {rmse_tri:.4f}")

# Ways to measure performance based on market or other model metrics:
# 1. If you have actual market_option_prices (e.g., from data file or API), compute:
#    mae = np.mean(np.abs(model_prices - market_option_prices))
#    rmse = np.sqrt(np.mean((model_prices - market_option_prices)**2))
#    relative_error = np.mean(np.abs((model_prices - market_option_prices) / market_option_prices))
# 2. Implied volatility error: Compute implied vols from model and market, then average absolute difference or RMSE.
# 3. Calibration error: Minimize sum of squared errors between model and market by optimizing parameters (e.g., via scipy.optimize).
# 4. Convergence rate: As shown above, track error vs n_steps; trinomial typically converges faster than binomial.
# 5. Computational efficiency: Time the pricing functions for large n_steps or many strikes.
#    import time
#    start = time.time()
#    price = trinomial_price(...)  # or other
#    elapsed = time.time() - start
#    print(f"Time: {elapsed}")
# 6. Out-of-sample prediction: Fit model to some strikes/maturities, test on others.
# 7. Hedging performance: Simulate delta-hedging error over paths, but requires Monte Carlo simulation.
