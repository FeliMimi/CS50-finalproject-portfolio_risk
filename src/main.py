import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# user input TODO: implement later with real input
tickers = ["AMZN", "MSFT", "NVDA"]
weights = np.array([0.5, 0.3, 0.2])
simulations = 100000


# loading data
data = yf.download(tickers, period="5y", auto_adjust=False)['Adj Close']

# dcalculating aily returns
daily_returns = data.pct_change().dropna()

# calculating statistics
mu = daily_returns.mean()
cov_matrix = daily_returns.cov()

# monte-carlo-simulation
simulated_returns = np.random.multivariate_normal(mu, cov_matrix, simulations)
portfolio_returns = simulated_returns @ weights

# risk numbers
mean_return = np.mean(portfolio_returns)
volatility = np.std(portfolio_returns)
VaR_5 = np.percentile(portfolio_returns, 5)
CVaR_5 = portfolio_returns[portfolio_returns <= VaR_5].mean()

print(f"Expected returns (daily): {mean_return:.4%}")
print(f"Volatility (daily): {volatility:.4%}")
print(f"5%-VaR (daily): {VaR_5:.4%}")
print(f"5%-CVaR (daily): {CVaR_5:.4%}")

# visualizing
plt.hist(portfolio_returns, bins=100, color='skyblue')
plt.axvline(VaR_5, color='red', linestyle='--', label="5%-VaR")
plt.title("Monte Carlo Portfolio Simulation")
plt.xlabel("Tägliche Rendite")
plt.ylabel("Häufigkeit")
plt.legend()
plt.show()

