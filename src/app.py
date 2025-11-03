from flask import Flask, render_template, request

import simulation # Import the simulation module

app = Flask(__name__) 

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        tickers = request.form.get('tickers').split(',')
        weights = [float(w) for w in request.form.get('weights').split(',')]
        n_sims = int(request.form.get('n_sims'))
        T_days = int(request.form.get('T_days'))
        
        # run simulation
        results, plots = simulation.run_simulation(tickers, weights, n_sims, T_days)
    
        return render_template('results.html', result=results, plots=plots)
   
    # Fallback to home if not POST
    return render_template('index.html')
