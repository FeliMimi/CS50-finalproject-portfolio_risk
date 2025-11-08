from flask import Flask, render_template, request, redirect, flash, url_for

import simulation # Import the simulation module

app = Flask(__name__) 
app.secret_key = "super secret string"

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            tickers = request.form.getlist('tickers')
            weights = [float(w) for w in request.form.getlist('weights')]
            n_sims = int(request.form.get('n_sims', 1000))
            T_days = int(request.form.get('days', 252))
            
            # control inputs
            if len(tickers) != len(weights):
                apology = "Number of tickers must match number of weights."
                return render_template("apology.html", apology=apology)
            
            if abs(sum(weights) - 1.0) > 1e-5:
                apology = "Weights must sum to 1."
                return render_template("apology.html", apology=apology)
            
            if n_sims <= 0 or T_days <= 0:
                apology = "Number of simulations and days must be positive integers."
                return render_template("apology.html", apology=apology)
            
            if not tickers:
                apology = "At least one ticker must be provided."
                return render_template("apology.html", apology=apology)
            
            if any(w < 0 for w in weights):
                apology = "Weights must be non-negative."
                return render_template("apology.html", apology=apology)
            

            # run simulation
            results, plots, mean_ret, median_ret, std_ret = simulation.run_simulation(
                tickers=tickers, 
                weights=weights, 
                n_sims=n_sims, 
                T_days=T_days
                )
            return render_template('results.html', results=results, plots=plots,
                                    mean_ret=mean_ret, std_ret=std_ret, median_ret=median_ret,
                                    n_sims=n_sims, T_days=T_days) 
        
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return render_template('apology.html', apology=str(e))
            
    # Fallback to home if not POST
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
