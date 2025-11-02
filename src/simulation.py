from datetime import datetime
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# user input TODO: implement later with real input and simulation parameters
tickers = ["AAPL", "MSFT", "NVDA"]
weights = np.array([0.5, 0.3, 0.2])
n_sims = 30_000
T_days = 50
dt = 1.0/252    # time step in days assuming 252 trading days
seed = 42
var_levels = [0.95, 0.99]   # 95% and 99% VaR

# set start end end date for market data request
end_date = datetime(year=2025, month=1, day=1)
start_date = datetime(year=2023, month=1, day=1)

# loading adjusted close prices
data = yf.download(tickers, start=start_date, end=end_date, interval="1d", auto_adjust=False)['Adj Close']

# getting the necessery parameters
log_rets = np.log(data / data.shift(1)).dropna()
mu = log_rets.mean().values
cov_matrix = log_rets.cov().values
variance = np.diag(cov_matrix)

# Cholesky decomposition
L = np.linalg.cholesky(cov_matrix)

# simulation settings
np.random.seed(seed)
n_assets = len(tickers)
S0 = data.iloc[-1].values

# array to hold portfolio value at horizon for each simulation
porfolio_end_values = np.zeros(n_sims)

# store a few example paths to plot
plot_sample_paths = 100
example_paths = np.zeros((plot_sample_paths, T_days +1, n_assets))
example_portfolio_paths = np.zeros((plot_sample_paths, T_days + 1))


# RUN MONTE CARLO SIMULATION
for sim in range(n_sims):
    # initialize prices
    prices = np.zeros((T_days +1, n_assets))
    prices[0] = S0.copy()

    # simulate day by day
    for t in range(1, T_days +1):
        Z = np.random.normal(size=n_assets)
        correlated_Z = L @ Z * np.sqrt(dt)

        # GBM step on log scale:
        # S_t = S_{t-1} * exp( (mu - 0.5*diag(cov)) * dt + correlated )
        drift = ( mu - 0.5 * np.diag(cov_matrix)) * dt
        prices[t] = prices[t-1] * np.exp(drift + correlated_Z)

    # computing final results
    final_prices = prices[-1]
    asset_returns = (final_prices / S0) - 1.0
    portfolio_return = np.dot(weights, asset_returns)
    porfolio_end_values[sim] = 1.0 + portfolio_return

    # save example paths
    if sim < plot_sample_paths:
        example_paths[sim] = prices
        example_portfolio_paths[sim] = np.sum(prices * weights, axis=1)

# ANALYZE RESULTS: DISTRIBUTION, VaR, ES

portfolio_returns = porfolio_end_values - 1.0

# compute VaR and ES
results = {}
for alpha in var_levels:
    losses = - portfolio_returns
    var_quantile = np.quantile(losses, 1 - alpha)
    tail_losses = losses[losses >= var_quantile]
    es = tail_losses.mean() if len(tail_losses) > 0 else np.nan

    results[alpha] = {'VaR': var_quantile, 'ES': es, 'num_tail_obs': len(tail_losses)}

# compute summary stats
mean_ret = portfolio_returns.mean()
median_ret = np.median(portfolio_returns)
std_ret = portfolio_returns.std()

# OUTPUT
print("Portfolio horizon (days):", T_days)
print("Number of simulations:", n_sims)
print("Portfolio return summary (simple returns over horizon):")
print(f" mean = {mean_ret:.4%}, median = {median_ret:.4%}, std = {std_ret:.4%}")
for alpha, res in results.items():
    print(f"{int(alpha*100)}% VaR (loss): {res['VaR']:.4%}, ES: {res['ES']:.4%}, tail observations: {res['num_tail_obs']}")

# PLOT
plt.figure(figsize=(9,5))
plt.hist(portfolio_returns, bins=200, density=False)
plt.title(f"Histogram of Portfolio Returns over {T_days} trading days")
plt.xlabel("Simple return (fraction)")
plt.ylabel("Frequency")
plt.grid(True)

# overlay VaR lines
for alpha, res in results.items():
    plt.axvline(x=-res['VaR'], linestyle='--', label=f"{int(alpha*100)}% VaR = {-res['VaR']:.2%}")
plt.legend()
plt.show()

# example paths for few simulations
plt.figure(figsize=(10,6))
for sim in range(plot_sample_paths):
    plt.plot(example_portfolio_paths[sim], alpha=0.7)
plt.title("Example simulated portfolio paths (first few sims)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.grid(True)
plt.show()
