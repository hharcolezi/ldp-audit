# General imports
import warnings; warnings.simplefilter('ignore')
from numba.core.errors import NumbaExperimentalFeatureWarning
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import xxhash
from scipy.optimize import minimize_scalar
from collections import defaultdict
import ray

# Import LDP protocols (by default from multi-freq-ldpy package -- https://github.com/hharcolezi/multi-freq-ldpy)
from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client
from multi_freq_ldpy.pure_frequency_oracles.UE import UE_Client
from multi_freq_ldpy.pure_frequency_oracles.SS import SS_Client
from multi_freq_ldpy.pure_frequency_oracles.LH import LH_Client
from multi_freq_ldpy.pure_frequency_oracles.HE import HE_Client

# Our imports
from .utils import setting_seed, find_tresh
from .base_auditor import LDPAuditor
from .approximate_ldp import find_scale, GM_Client

class LongitudinalLDPAuditor(LDPAuditor):
    """
    The LongitudinalLDPAuditor class audits various Local Differential Privacy (LDP) protocols 
    over multiple data collections (i.e., longitudinal data).
    """

    def __init__(self, nb_trials: int = int(1e6), alpha: float = 1e-2, 
                 epsilon: float = 0.25, delta: float = 0.0, k: int = 2,
                 random_state: int = 42, n_jobs: int = -1, tau: int = 2):
        """
        Initializes the LongitudinalLDPAuditor with the specified parameters.

        Parameters:
        ----------
        nb_trials : int
            The number of trials for the audit (default is 1e6).
        alpha : float
            The significance level for the Clopper-Pearson confidence intervals (default is 1e-2).
        epsilon : float
            The theoretical privacy budget (default is 0.25).
        delta : float
            The privacy parameter for approximate LDP protocols (default is 0.0 -- pure LDP).
        k : int
            The domain size (default is 2).
        random_state : int
            The random seed for reproducibility (default is 42).
        n_jobs : int
            The number of CPU cores to use for parallel processing (-1 uses all available cores, default is None).
        tau : int
            The number of data collections (default is 2).
        """
        super().__init__(nb_trials, alpha, epsilon, delta, k, random_state, n_jobs)
        self.tau = tau

        # possible protocols to audit
        self.protocols = {

            # pure LDP protocols
            "GRR": self.audit_grr,
            "SS":  self.audit_ss,
            "SUE": self.audit_sue,
            "OUE": self.audit_oue,
            "THE": self.audit_the,
            "SHE": self.audit_she,
            "BLH": self.audit_blh,
            "OLH": self.audit_olh,

            # approximate LDP protocols
            "AGM": self.audit_agm,
            "GM": self.audit_gm,
        }

    def get_params(self):
        """
        Get the parameters of the LongitudinalLDPAuditor.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params()
        params['tau'] = self.tau
        return params

    def set_params(self, **params):
        """
        Set the parameters of the LongitudinalLDPAuditor.

        Parameters:
        ----------
        **params : dict
            Auditor parameters.

        Returns:
        -------
        self : auditor instance
            Auditor instance.
        """
        if 'tau' in params:
            self.tau = params.pop('tau')
        return super().set_params(**params)

    #================================= Audit Methods for Approximate LDP Protocols =================================
    @ray.remote
    def audit_grr(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Generalized Randomized Response (GRR) protocol for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        
        # Suppress specific Numba warnings
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)

        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0    
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                rep[GRR_Client(v, k, epsilon)] += 1
            
            m = max(rep)        
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic
        
        return count

    @ray.remote
    def audit_ss(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Subset Selection (SS) protocol for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)

        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                rep[SS_Client(v, k, epsilon)] += 1            
        
            m = max(rep)
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic

        return count

    @ray.remote
    def audit_sue(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Symmetric Unary Encoding (SUE -- a.k.a. RAPPOR) protocol for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                rep += UE_Client(v, k, epsilon, False)
        
            m = max(rep)
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic

        return count

    @ray.remote
    def audit_oue(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Optimized Unary Encoding (OUE) protocol for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                rep += UE_Client(v, k, epsilon, True)
        
            m = max(rep)
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic

        return count

    @ray.remote
    def audit_the(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Thresholding with Histogram Encoding (THE) protocol for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        thresh = minimize_scalar(find_tresh, bounds=[0.5, 1], method='bounded', args=(epsilon)).x 
        count = 0
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                rep[np.where(HE_Client(v, k, epsilon) > thresh)[0]] += 1   
                        
            m = max(rep)
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic

        return count

    @ray.remote
    def audit_she(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Summation with Histogram Encoding (SHE) protocol for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """

        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)

        setting_seed(random_state)
        np.random.seed(random_state)
        
        count = 0
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                rep += HE_Client(v, k, epsilon)
                
            m = max(rep) 
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic
        
        return count

    @ray.remote
    def audit_blh(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Binary Local Hashing (BLH) protocol for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        g = 2
        count = 0
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                lh_val, rnd_seed = LH_Client(v, k, epsilon, False)
                
                ss_lh = []
                for val in range(k):
                    if lh_val == (xxhash.xxh32(str(val), seed=rnd_seed).intdigest() % g):
                        ss_lh.append(val)
                
                rep[ss_lh] += 1
                
            m = max(rep)
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic

        return count
    
    @ray.remote
    def audit_olh(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Optimal Local Hashing (OLH) protocol for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP (not used for pure LDP).
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        
        setting_seed(random_state)
        np.random.seed(random_state)

        g = int(np.round(np.exp(epsilon))) + 1
        count = 0
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                lh_val, rnd_seed = LH_Client(v, k, epsilon, True)
                
                ss_lh = []
                for val in range(k):
                    if lh_val == (xxhash.xxh32(str(val), seed=rnd_seed).intdigest() % g):
                        ss_lh.append(val)
                
                rep[ss_lh] += 1
                
            m = max(rep)
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic

        return count

    #================================= Audit Methods for Approximate LDP Protocols =================================
    @ray.remote
    def audit_agm(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Analytical Gaussian Mechanism (AGM) for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP.
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        setting_seed(random_state)
        np.random.seed(random_state)
        
        count = 0
        Delta_2 = np.sqrt(2)
        sigma = find_scale(epsilon, delta, Delta_2)
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                rep += GM_Client(v, k, sigma)

            m = max(rep) 
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic

        return count

    @ray.remote
    def audit_gm(self, random_state, trials, v, k, epsilon, delta, test_statistic):
        """
        Audits the Gaussian Mechanism (GM) for longitudinal data.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : int
            True value.
        k : int
            Domain size.
        epsilon : float
            Theoretical privacy budget.
        delta : float
            Privacy parameter for approximate LDP.
        test_statistic : int
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences based on the test statistic.
        """
        
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        setting_seed(random_state)
        np.random.seed(random_state)
        
        count = 0
        Delta_2 = np.sqrt(2)
        sigma = (Delta_2 / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
        for _ in range(trials):
            rep = np.zeros(k)
            for t in range(self.tau):
                rep += GM_Client(v, k, sigma)

            m = max(rep) 
            count += np.random.choice(np.where(rep == m)[0]) == test_statistic

        return count

    #================================= General Audit Method =================================

    def run_audit(self, protocol_name):
        """
        Runs the audit for the specified LDP protocol over longitudinal data.

        Parameters:
        ----------
        protocol_name : str
            The name of the LDP protocol to audit.

        Returns:
        -------
        float
            The estimated empirical epsilon value.

        Raises:
        ------
        ValueError
            If the specified protocol name is not supported.
        """
        if protocol_name not in self.protocols:
            raise ValueError(f"Unsupported protocol: {protocol_name}")

        protocol = self.protocols[protocol_name]

        TP, FP = [], []  # initialize list for parallelized results
        for idx in range(self.nb_cores):
            unique_seed = xxhash.xxh32(protocol_name).intdigest() + self.random_state + idx
            TP.append(protocol.remote(self, unique_seed, self.lst_trial_per_core[idx], self.v1, self.k, self.epsilon, self.delta, self.v1))
            FP.append(protocol.remote(self, unique_seed, self.lst_trial_per_core[idx], self.v2, self.k, self.epsilon, self.delta, self.v1))

        # take results with get function from Ray library
        TP = sum(ray.get(TP))
        FP = sum(ray.get(FP))

        # Clopper-Pearson confidence intervals
        TPR = proportion_confint(TP, self.nb_trials, self.alpha / 2, 'beta')[0]
        FPR = proportion_confint(FP, self.nb_trials, self.alpha / 2, 'beta')[1]

        # empirical epsilon estimation 
        self.eps_emp = np.log((TPR - self.delta) / FPR)

        if self.eps_emp > self.epsilon:
            warnings.warn(f"Empirical epsilon ({self.eps_emp}) exceeds theoretical epsilon ({self.epsilon}). There might be an error in the LDP-Auditor code or the LDP protocol being audited is wrong.")
        
        return self.eps_emp



# # Example
# seed = 42
# nb_trials = int(1e6)
# alpha = 1e-2
# epsilon = 1
# k = 2
# tau = 25

# print('=====Auditing pure LDP protocols=====')
# delta = 0.0
# pure_ldp_protocols = ['GRR', 'SS', 'SUE', 'OUE', 'THE', 'SHE', 'BLH', 'OLH']
# auditor_pure_ldp = LongitudinalLDPAuditor(nb_trials=nb_trials, alpha=alpha, epsilon=epsilon, delta=delta, k=k, random_state=seed, n_jobs=-1, tau=tau)

# for protocol in pure_ldp_protocols:
#     eps_emp = auditor_pure_ldp.run_audit(protocol)
#     print("{} eps_emp:".format(protocol), eps_emp)


# print('\n=====Auditing approximate LDP protocols=====')
# delta = 1e-5
# approx_ldp_protocols = ['AGM', 'GM']
# auditor_approx_ldp = LongitudinalLDPAuditor(nb_trials=nb_trials, alpha=alpha, epsilon=epsilon, delta=delta, k=k, random_state=seed, n_jobs=-1, tau=tau)

# for protocol in approx_ldp_protocols:
#     eps_emp = auditor_approx_ldp.run_audit(protocol)
#     print("{} eps_emp:".format(protocol), eps_emp)




