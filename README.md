# CS50 Final Project - Portfolio Risk Analysis
#### Video Demo: https://youtu.be/4lIzGaP_ivY
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

The simulation itself uses **Geometric Brownian Motion (GBM)** to calculate a number of possible future developments for the portfolio. The GBM formula is applied within two nested for-loops:  
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

- `layout.html`: Defines the basic design of the web application. A **.css** file and **Bootstrap** are included. Other design elements include a header and a footer for the website.

- `index.html`: This is the homepage, where users can enter their portfolio details and parameters for the simulation. The button to add another stock is connected to JavaScript code, allowing dynamic additions. User input is validated **client-side** to provide a more user-friendly experience.

- `results.html`: Displays the simulation results. At the top, a disclaimer informs users that the results should not be used for real investments, as they are based on simplified models and assumptions.  
  Mean return, median return, and standard deviation are shown as text.  
  VaR, ES, and the number of tail observations are presented in a table for clarity.  
  Below the table, two plots are displayed as Base64 strings.  
  A button is provided to return to the homepage.

- `apology.html`: Displays an error message when something goes wrong during the simulation. A button redirects users back to the homepage.


#### static

A small **.css** file used for styling the web application. It was largely written with the help of ChatGPT and GitHub Copilot.  

It defines general layout, typography, form inputs, buttons, and the results page. Features include responsive input fields, styled submit and add buttons, containers with shadows, and basic flexbox layouts for plots and stock entry fields. The design aims for a clean, user-friendly interface and is kept pretty minimalistic.

#### Other Files

The project also includes a `requirements.txt` file, which lists all the modules necessary to run the program.  
Additionally, a `.gitignore` file is included to ensure that unnecessary files are not pushed to Git.


