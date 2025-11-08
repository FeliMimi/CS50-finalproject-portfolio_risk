# CS50 Final Project - Portfolio Risk Analysis
#### Video Demo:  <URL HERE>
#### Description:

This project evaluates the risk of an investment portfolio using Monte Carlo methods.
The simulation itself is implemented in python. To make it more interactible and User-friendly I built a simple web-application where a user can enter his portfolio details and the simulation parameters. The results are then displayed in both a table and in two plots.

##### simulation.py
This is the main program behind the whole project.
It basically is a function that is then used in `app.py`. The function takes some parameters that are hardcoded (`dt`, `var_levels`, `seed`) and some parameters (`tickers`, `weights`, `n_sims`, `T_days`) that depent on user input.
It loads historical finance data using the [`yfinance`](https://pypi.org/project/yfinance/) module in Python. Then it calculates the relevant parameters necessary for the simulation. The simulation itself uses, Geometric-Brownian-Motion (GBM) to calculate a given number of paths. With some math we can simulate many possible future returns for the portfolio. This is done through two nested `for`-loops, which currently limit execution speed.  
   (One of these loops could be **vectorized** using NumPy to significantly improve efficiency.)
The simulation results are then analyzed and used to compute several **key performance indicators(KPIs)** commonly used in financial analysis (e.g. Value at Risk, Expected Shortfall). 
The function also generates two plots. The first one is a histogram that displays the portfolio returns with overlayed **Value at Risk** lines. The second one is visually interesting. It shows some example paths for the portfolio returns. These plots are converted to **Base64-encoded strings'' so that they can be rendered directly on the web interface.
The function returns the two generated plots (as Base64 strings) and the computed financial key metrics that were calculated in the simulation.


##### app.py
This file handles all the flask related stuff. It combines `simulation.py` and all the relevant `.html` files. Only a single route is needed because we 