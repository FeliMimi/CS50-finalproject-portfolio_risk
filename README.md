# CS50 Final Project - Portfolio Risk Analysis
#### Video Demo:  <URL HERE>
#### Description:

This project evaluates the risk of an investment portfolio using Monte Carlo methods.
The simulation itself is implemented in python. To make it more interactible and User-friendly I built a simple web-application where a user can enter his portfolio details and the simulation parameters. The results are then displayed in both a table and in two plots.

##### simulation.py
This is the main program behind the whole project.
It basically is a function that is then used in `app.py`. The function takes some parameters that are hardcoded (`dt`, `var_levels`, `seed`) and some parameters (`tickers`, `weights`, `n_sims`, `T_days`) that depent on user input.
It loads historical finance data using the [`yfinance`](https://pypi.org/project/yfinance/) module in Python. Then it calculates the relevant parameters necessary for the simulation. The simulation itself uses, Geometric-Brownian-Motion to calculate a given number of paths. With some math we can simulate many possible future returns for the portfolio. This is done through two nested `for`-loops, which currently limit execution speed.  
   (One of these loops could be **vectorized** using NumPy to significantly improve efficiency.)
The simulation results are then analyzed and used to compute some common key performance indicators for financial analysis. 
The function also produces two plots. The first one is a histogram that displays the portfolio returns with an overlay of Value at Risk lines. The second one is visually interesting. It shows some example paths for the portfolio returns. These plots are converted to base64 to display them on the website.
The function returns the plots and the financial key parameters that were calculated in the simulation.
