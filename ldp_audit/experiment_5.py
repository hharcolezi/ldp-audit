# general imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging

# Our imports
from ldp_audit.multidimensional_auditor import MultidimensionalLDPAuditor
from plot_functions import plot_results_multidimensional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Case Study #4: LDP Auditing with Multidimensional Data

def run_multidimensional_experiments(nb_trials: int, alpha: float, lst_protocols: list, lst_seed: list, uni_k: list, lst_eps: list, lst_d: list, delta: float, analysis: str):
    
    # Initialize dictionary to save results
    results = {
        'seed': [],
        'protocol': [],
        'k': [],
        'd': [],
        'delta': [],
        'epsilon': [],    
        'eps_emp': []
    }

    # Initialize LDP-Auditor
    auditor = MultidimensionalLDPAuditor(nb_trials=nb_trials, alpha=alpha, epsilon=lst_eps[0], delta=delta, k=uni_k[0], random_state=lst_seed[0], n_jobs=-1, d=lst_d[0])

    # Run the experiments
    for seed in lst_seed:    
        logging.info(f"Running experiments for seed: {seed}")
        for k in uni_k:
            for d in lst_d:            
                for epsilon in lst_eps:

                    # Update the auditor parameters
                    auditor.set_params(epsilon=epsilon, k=k, random_state=seed, d=d)

                    for protocol in tqdm(lst_protocols, desc=f'seed={seed}, k={k}, d={d}, epsilon={epsilon}, d={d}'):
                        eps_emp = auditor.run_audit(protocol)                    
                        results['seed'].append(seed)
                        results['protocol'].append(protocol)
                        results['k'].append(k)
                        results['d'].append(d)
                        results['delta'].append(delta)
                        results['epsilon'].append(epsilon)
                        results['eps_emp'].append(eps_emp)

    # Convert and save results in csv file
    df = pd.DataFrame(results)
    df.to_csv(f'results/ldp_audit_results_{analysis}.csv', index=False)
    
    return df

if __name__ == "__main__":
    ## General parameters
    analysis_multidimensional = 'multidimensional_rspfd'
    lst_protocol_mdim = ["RS+FD[GRR]", "RS+FD[SUE-z]", "RS+FD[SUE-r]", "RS+FD[OUE-z]", "RS+FD[OUE-r]"]
    lst_eps = [0.25, 0.5, 0.75, 1, 2, 4, 6, 10]
    lst_d = [2, 10]
    uni_k = [2, 100] # appendix -> [25, 50]
    lst_seed = range(5)
    nb_trials = int(1e6)
    alpha = 1e-2
    delta = 0

    df_multidimensional = run_multidimensional_experiments(nb_trials, alpha, lst_protocol_mdim, lst_seed, uni_k, lst_eps, lst_d, delta, analysis_multidimensional)
    plot_results_multidimensional(df_multidimensional, lst_protocol_mdim, analysis_multidimensional, lst_eps, uni_k, lst_d) # Main results -- Figure 7 in paper