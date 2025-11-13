# CS50 Final Project - Portfolio Risk Analysis
#### Video Demo:  <URL HERE>
### Description:

This project evaluates the **risk of an investment portfolio** using **Monte Carlo simulation methods**.  
The core simulation is implemented in **Python**, where historical market data is analyzed to estimate potential future portfolio outcomes.  

To make the project more interactive and user-friendly, I developed a **web application** that allows users to enter their portfolio details (stock tickers and weights) as well as simulation parameters (number of simulations and time horizon).  
Once the input is submitted, the program runs the simulation and presents the results directly on the website.  

The output includes a **table** summarizing key financial metrics — such as Value at Risk (VaR) and Expected Shortfall — along with **two plots** that visualize the results.  
One plot shows the distribution of simulated returns with VaR levels highlighted, while the other displays several example paths of how the portfolio value might evolve over time.  

Overall, the project combines **data analysis**, **financial modeling**, and **web development** to provide an accessible way to explore and understand portfolio risk.


#### simulation.py

This is the main program behind the whole project.  
It defines a function that is later used in `app.py` to run the Monte Carlo simulation. The function takes some parameters that are hardcoded (`var_levels`, `seed`) and some parameters that depend on user input (`tickers`, `weights`, `n_sims`, `T_days`).

The function loads historical financial data using the [`yfinance`](https://pypi.org/project/yfinance/) module in Python. To limit runtime, only data from the last 4 years is downloaded. Then, the daily **logarithmic returns** are calculated from the data. From these returns, the relevant parameters needed for the simulation (mean and variance) are computed. The simulation also requires other parameters such as a random seed, which is generated with the `np.random()` function.

Next, several arrays are initialized to store the simulation results. For visualization purposes, around 100 example paths from the simulation are stored separately.  

The simulation itself uses **Geometric Brownian Motion (GBM)** to calculate a number of possible future developments for the portfolio. The GBM formula is applied within two nested `for` loops:  
one iterates over each day (`T_days`), and the other over the number of simulations provided by the user (`n_sims`). This double loop currently limits execution speed — one of these loops could be **vectorized** using NumPy to significantly improve efficiency.
However, this was not implemented to keep the codebase simpler and easier to understand within the scope of CS50.

The results are then stored and analyzed to compute several **key performance indicators (KPIs)** commonly used in financial analysis (e.g. Value at Risk, Expected Shortfall).  
The function also generates two plots:  
the first is a histogram displaying portfolio returns with overlaid **Value at Risk** lines, and the second shows several example simulation paths for the portfolio’s future value.  

These plots are converted into **Base64-encoded strings** so they can be displayed directly on the web interface.

Finally, the function returns both generated plots (as Base64 strings) and the computed financial key metrics obtained from the simulation.


#### app.py

This file handles all Flask-related functionality.  
It connects `simulation.py` with the relevant `.html` templates to create the web interface. Only a single route is needed, since all templates can be loaded from the `index.html` file. The `simulation` module is imported so that its main function can be used directly within `app.py`.

In the `index` function, the user’s input from the web form is processed. The input is validated on the **server side** to prevent errors during the simulation run.  
If the user provides invalid input, a template called `apology.html` is rendered, displaying an appropriate error message.

If the input is valid, the simulation is executed using the provided parameters, and the `results.html` template is rendered to display the outcomes.

If the route is accessed via a method other than `POST`, the homepage (`index.html`) is rendered by default.


#### templates

There are four `.html` templates in the project:
   - `layout.html`: Here the basic design of the Web-application is defined. Both a **.css** file and **Bootstrap** is included.
   Some other design features include a header and footnote for the website.
   - `index-html`: This is the homepage, where the user can enter his portfolio details and the parameters for the simulation. The button to add another stock is conected to some JavaScript code so it can be done dynamically. The users input is here controlled **client side** as to provide a more user-friendly experience.
   - `results.html`: Here the results of the simualtion are displayed. BUt firat of all there is a disclaimer, that the results should not be used for real investements, because they are calculated on simplified models and assumptions. 
   Mean return, median return and standard deviation are displayed as text.
   VaR, ES and the number of tail observations are displayed in a table for better clarity.
   Below that the two plots are displayed as Base64-strings.
   There is also a button to return to the homepage.