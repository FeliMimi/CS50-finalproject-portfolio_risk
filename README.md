# CS50 Final Project - Portfolio Risk Analysis

## Description
This project evaluates the risk of an investment portfolio using Monte Carlo methods. It loads historical finance data using the `yfinance` module in Python. Then it calculates the relevant parameters necessery for the analysis. These parameters are used, to calculate a given number of paths using Geometric-Brownian-Motion. With this large number of possible portfolio-returns one can evaluate the risk his portfolio has. The most common parameter to analize risk is implemeted. VaR (Value at Risk) at an intervall of both 95% and 99%.

WEbpage.......

## Technologies
- Python
- panda, Numpy, Matplotlib
- yfinance (for stock data)

## Planned Features
1. Load historical stock data
2. Calculate portfolio returns
3. Run Monte Carlo simulations
4. Compute VaR and CVaR
5. Visualize results
