# general imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging

# Our imports
from ldp_audit.base_auditor import LDPAuditor
from plot_functions import plot_result_pure_ldp_package

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Case Study #5:  Debugging a Python Implementation of UE Protocols

def run_pure_ldp_pck_experiments(nb_trials: int, alpha: float, lst_protocols: list, lst_seed: list, lst_k: list, lst_eps: list, delta: float, analysis: str):
    
    # Initialize dictionary to save results
    results = {
            'seed': [],
            'protocol': [],
            'k': [],
            'delta': [],
            'epsilon': [],    
            'eps_emp': []
            }

    # Initialize LDP-Auditor
    auditor = LDPAuditor(nb_trials=nb_trials, alpha=alpha, epsilon=lst_eps[0], delta=delta, k=lst_k[0], random_state=lst_seed[0], n_jobs=-1)

    # Run the experiments
    for seed in lst_seed:    
        logging.info(f"Running experiments for seed: {seed}")
        for k in lst_k:
            for epsilon in lst_eps:
                
                # Update the auditor parameters
                auditor.set_params(epsilon=epsilon, k=k, random_state=seed)

                for protocol in tqdm(lst_protocols, desc=f'seed={seed}, k={k}, epsilon={epsilon}'):    
                    eps_emp = auditor.run_audit(protocol)                    
                    results['seed'].append(seed)
                    results['protocol'].append(protocol)
                    results['k'].append(k)
                    results['delta'].append(delta)
                    results['epsilon'].append(epsilon)
                    results['eps_emp'].append(eps_emp)

    # Convert and save results in csv file
    df = pd.DataFrame(results)
    df.to_csv('results/ldp_audit_results_{}.csv'.format(analysis), index=False)
    
    return df

if __name__ == "__main__":
    ## General parameters
    analysis_pure_ldp_pck = 'pure_ldp_package'
    pure_ldp_pck_protocols = ['SUE_pure_ldp_pck', 'OUE_pure_ldp_pck']
    lst_eps = [0.25, 0.5, 0.75, 1, 2, 4, 6, 10]
    lst_k = [25, 50, 100, 150, 200]
    lst_seed = range(5)
    nb_trials = int(1e6)
    alpha = 1e-2
    delta = 0.0 
    
    df_pure_ldp_pck = run_pure_ldp_pck_experiments(nb_trials, alpha, pure_ldp_pck_protocols, lst_seed, lst_k, lst_eps, delta, analysis_pure_ldp_pck)
    plot_result_pure_ldp_package(df_pure_ldp_pck, analysis_pure_ldp_pck, pure_ldp_pck_protocols, lst_eps, lst_k) # Main results -- Figure 8 in paper