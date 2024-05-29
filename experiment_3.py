# general imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging

# Our imports
from ldp_audit.lho_auditor import LHOAuditor
from plot_functions import plot_results_lho_protocol

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Case Study #2: Auditing the Privacy Loss of Local Hashing Encoding Without LDP

def run_lho_experiments(nb_trials: int, alpha: float, lst_seed: list, lst_k: list, lst_g: list, analysis: str):
    
    # Initialize dictionary to save results
    results = {
        'seed': [],
        'g': [],
        'k': [],
        'eps_emp': []
    }

    # Initialize LHO-Auditor
    auditor = LHOAuditor(nb_trials=nb_trials, alpha=alpha, k=lst_k[0], random_state=lst_seed[0], n_jobs=-1, g=lst_g[0])

    # Run the experiments
    for seed in lst_seed:    
        logging.info(f"Running experiments for seed: {seed}")
        for k in lst_k:
            for g in tqdm(lst_g, desc=f'seed={seed}, k={k}'):

                # Update the auditor parameters
                auditor.set_params(random_state=seed, k=k, g=g)
                
                # Run audit for LHO protocol
                eps_emp = auditor.run_audit("LHO")
                
                # Save the results
                results['seed'].append(seed)
                results['g'].append(g)
                results['k'].append(k)
                results['eps_emp'].append(eps_emp)

    # Convert and save results in csv file
    df = pd.DataFrame(results)
    df.to_csv(f'results/ldp_audit_results_{analysis}.csv', index=False)
    
    return df

if __name__ == "__main__":
    ## General parameters
    analysis_lho = 'lho_no_ldp'
    lst_k = [25, 50, 100, 150, 200]
    lst_seed = range(5)
    lst_g = range(2, 11)
    nb_trials = int(1e6)
    alpha = 1e-2

    df_lho = run_lho_experiments(nb_trials, alpha, lst_seed, lst_k, lst_g, analysis_lho)
    plot_results_lho_protocol(df_lho, analysis_lho, lst_k, lst_g) # Main results -- Figure 5 in paper    