# general imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging

# Our imports
from ldp_audit.base_auditor import LDPAuditor
from plot_functions import plot_results_approx_ldp_delta_impact

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Case Study #1: Approximate- VS Pure-LDP

def run_delta_imp_experiments(nb_trials: int, alpha: float, lst_protocols: list, lst_seed: list, lst_k: list, lst_eps: list, lst_delta: list, analysis: str):
    
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
    auditor = LDPAuditor(nb_trials=nb_trials, alpha=alpha, epsilon=lst_eps[0], delta=lst_delta[0], k=lst_k[0], random_state=lst_seed[0], n_jobs=-1)

    # Run the experiments
    for seed in lst_seed:    
        logging.info(f"Running experiments for seed: {seed}")
        for k in lst_k:
            for epsilon in lst_eps:
                for delta in lst_delta:
                
                    # Update the auditor parameters
                    auditor.set_params(epsilon=epsilon, delta=delta, k=k, random_state=seed)

                    for protocol in tqdm(lst_protocols, desc=f'seed={seed}, delta={delta}, k={k}, epsilon={epsilon}'):    
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
    analysis_delta_impact = 'approximate_ldp_delta_impact'
    approx_ldp_protocols = ['AGRR', 'ASUE', 'ABLH', 'AOLH', 'GM', 'AGM']
    lst_eps = [0.25, 0.5, 0.75, 1]
    lst_k = [25, 200] # appendix -> [100, 150]
    lst_seed = range(5)
    lst_delta = [1e-7, 1e-6, 1e-4] # 1e-5 already executed in main experiments, we'll merge them on the plotting step
    nb_trials = int(1e6)
    alpha = 1e-2

    df_delta_imp = run_delta_imp_experiments(nb_trials, alpha, approx_ldp_protocols, lst_seed, lst_k, lst_eps, lst_delta, analysis_delta_impact)
    plot_results_approx_ldp_delta_impact(df_delta_imp, analysis_delta_impact, approx_ldp_protocols, lst_eps, lst_k, sorted(lst_delta+[1e-5])) # Main results -- Figure 4 in paper