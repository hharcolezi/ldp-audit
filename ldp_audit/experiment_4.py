# general imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging
from statsmodels.stats.proportion import proportion_confint

# Our imports
from ldp_audit.longitudinal_auditor import LongitudinalLDPAuditor
from plot_functions import plot_results_longitudinal_pure_ldp_protocols

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Case Study #3: Auditing the LDP Sequential Composition in Longitudinal Studies

def run_longitudinal_experiments(nb_trials: int, alpha: float, lst_protocols: list, lst_seed: list, lst_k: list, lst_eps: list, lst_tau: list, delta: float, analysis: str):
    
    # Initialize dictionary to save results
    results = {
            'seed': [],
            'protocol': [],
            'k': [],
            'tau': [],
            'delta': [],
            'epsilon': [],    
            'eps_emp': []
            }

    # Initialize LDP-Auditor
    auditor = LongitudinalLDPAuditor(nb_trials=nb_trials, alpha=alpha, epsilon=lst_eps[0], delta=delta, k=lst_k[0], random_state=lst_seed[0], n_jobs=-1, tau=lst_tau[0])

    # Run the experiments
    for seed in lst_seed:    
        logging.info(f"Running experiments for seed: {seed}")
        for k in lst_k:
            for epsilon in lst_eps:
                for tau in lst_tau:

                    # Update the auditor parameters
                    auditor.set_params(epsilon=epsilon, k=k, random_state=seed, tau=tau)

                    for protocol in tqdm(lst_protocols, desc=f'seed={seed}, k={k}, epsilon={epsilon}, tau={tau}'):
                        eps_emp = auditor.run_audit(protocol)                    
                        results['seed'].append(seed)
                        results['protocol'].append(protocol)
                        results['k'].append(k)
                        results['tau'].append(tau)
                        results['delta'].append(delta)
                        results['epsilon'].append(epsilon)
                        results['eps_emp'].append(eps_emp)

    # Convert and save results in csv file
    df = pd.DataFrame(results)
    df.to_csv('results/bk_ldp_audit_results_{}.csv'.format(analysis), index=False)
    
    return df

if __name__ == "__main__":
    ## General parameters
    analysis_long = 'sequential_composition'
    lst_eps = [0.25, 0.5, 0.75, 1]
    lst_k = [2, 100] # appendix -> [25, 50]
    lst_seed = range(2)
    lst_tau = np.array([5, 10, 25, 50, 75, 100, 250, 500])
    nb_trials = int(1e6)
    alpha = 1e-2

    ## Calculate Monte Carlo upper bound (i.e., \epsilon_{OPT})
    CPL = proportion_confint(nb_trials, nb_trials, alpha/2, 'beta')[0]
    CPU = proportion_confint(0, nb_trials, alpha/2, 'beta')[1]
    eps_ub = np.log(CPL/CPU)

    ## pure LDP protocols
    pure_ldp_protocols = ['GRR', 'SS', 'SUE', 'OUE', 'THE', 'SHE', 'BLH', 'OLH']
    delta = 0.0
    analysis_pure_long = 'sequential_composition_pure_ldp_protocols'
    df_pure_long = run_longitudinal_experiments(nb_trials, alpha, pure_ldp_protocols, lst_seed, lst_k, lst_eps, lst_tau, delta, analysis_pure_long)

    ## approximate LDP protocols
    approx_lst_protocol = ["GM", "AGM"]
    delta = 1e-5
    analysis_approx_long = 'sequential_composition_approx_ldp_protocols'
    df_approx_long = run_longitudinal_experiments(nb_trials, alpha, approx_lst_protocol, lst_seed, lst_k, lst_eps, lst_tau, delta, analysis_approx_long)

    plot_results_longitudinal_pure_ldp_protocols(df_pure_long, df_approx_long, pure_ldp_protocols, approx_lst_protocol, analysis_long, lst_eps, lst_k, lst_tau, eps_ub) # Main results -- Figure 6 and in paper