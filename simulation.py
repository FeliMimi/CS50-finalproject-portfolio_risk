from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def run_simulation(tickers, weights, n_sims, T_days, dt=1.0/252, var_levels=[0.95, 0.99], seed=42):
    """Run Monte Carlo simulation for portfolio VaR and ES estimation.

    Parameters:
    tickers : list of str
        List of asset tickers.
    weights : np.array
        Portfolio weights corresponding to the tickers.
    n_sims : int
        Number of Monte Carlo simulations.
    T_days : int
        Time horizon in days.
    dt : float
        Time step in years (default is 1/252 for daily steps).
    var_levels : list of float
        List of VaR confidence levels (default [0.95, 0.99]).
    seed : int
        Random seed for reproducibility (default 42).

    Returns:
    results : dict
        Dictionary containing VaR, ES, and summary statistics.
    plots : list of str
        List of Base64 encoded plot images.
    mean_ret : float
        Mean portfolio return over the horizon.
    std_ret : float
        Standard deviation of portfolio return over the horizon.
    median_ret : float 
        Median portfolio return over the horizon.
    """
    # set start end end date for market data request
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*3)  # last 3 years

    # loading adjusted close prices
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d", auto_adjust=False)['Adj Close']

    # getting the necessery parameters
    log_rets = np.log(data / data.shift(1)).dropna()
    mu = log_rets.mean().values
    cov_matrix = log_rets.cov().values

    # Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)

    # simulation settings
    np.random.seed(seed)
    n_assets = len(tickers)
    S0 = data.iloc[-1].values

    # array to hold portfolio value at horizon for each simulation
    portfolio_end_values = np.zeros(n_sims)

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
        portfolio_end_values[sim] = 1.0 + portfolio_return

        # save example paths
        if sim < plot_sample_paths:
            example_paths[sim] = prices
            example_portfolio_paths[sim] = np.sum(prices * weights, axis=1)

    # ANALYZE RESULTS: DISTRIBUTION, VaR, ES

    portfolio_returns = portfolio_end_values - 1.0

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

    # function to convert plots to Base64 strings for HTML embedding
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        return image_base64


    # PLOT
    fig1, ax1 = plt.subplots(figsize=(9,5))
    ax1.hist(portfolio_returns, bins=200, density=False)
    ax1.set_title(f"Histogram of Portfolio Returns over {T_days} trading days")
    ax1.set_xlabel("Simple return (fraction)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True)

    # overlay VaR lines
    for alpha, res in results.items():
        ax1.axvline(x=-res['VaR'], linestyle='--', label=f"{int(alpha*100)}% VaR = {-res['VaR']:.2%}")
    ax1.legend()

    # convert to base64
    plot1_base64 = fig_to_base64(fig1)

    # example paths for few simulations
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for sim in range(plot_sample_paths):
        ax2.plot(example_portfolio_paths[sim], alpha=0.7)
    ax2.set_title("Example simulated portfolio paths (first 100 sims)")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price")
    ax2.grid(True)

    # convert to base64
    plot2_base64 = fig_to_base64(fig2)

    plots = [plot1_base64, plot2_base64]

    return results, plots, mean_ret, std_ret, median_ret