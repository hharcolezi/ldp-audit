# General imports
import warnings; warnings.simplefilter('ignore')
from numba.core.errors import NumbaExperimentalFeatureWarning
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from collections import defaultdict
import ray
import xxhash

# Import LDP protocols (by default from multi-freq-ldpy package -- https://github.com/hharcolezi/multi-freq-ldpy)
from multi_freq_ldpy.mdim_freq_est.RSpFD_solution import RSpFD_GRR_Client, RSpFD_UE_zero_Client, RSpFD_UE_rnd_Client

# Our imports
from .utils import setting_seed
from .base_auditor import LDPAuditor
from .attacks import attack_ue

class MultidimensionalLDPAuditor(LDPAuditor):
    """
    The MultidimensionalLDPAuditor class is designed to audit various Local Differential Privacy (LDP) protocols 
    for multidimensional data following the RS+FD solution.
    """

    def __init__(self, nb_trials: int = int(1e6), alpha: float = 1e-2, 
                 epsilon: float = 0.25, delta: float = 0.0, k: int = 2, 
                 random_state: int = 42, n_jobs: int = -1, d: int = 2):
        """
        Initializes the MultidimensionalLDPAuditor with the specified parameters.

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
        random_state : int
            The random seed for reproducibility (default is 42).
        n_jobs : int
            The number of CPU cores to use for parallel processing (-1 uses all available cores, default is None).
        d : int
            Number of attributes (dimensions).
        """
        super().__init__(nb_trials, alpha, epsilon, delta, k, random_state, n_jobs)
        self.d = d
        self.lst_k = [self.k for _ in range(self.d)] # List of domain sizes for each attribute.
        self.v1 = [0 for _ in range(self.d)]   # Update v1 for a multidimensional vector
        self.v2 = [self.k-1 for _ in range(self.d)] # Update v2 for a multidimensional vector

        # possible protocols to audit
        self.protocols = {

            # RS+FD LDP protocols
            "RS+FD[GRR]": self.audit_grr,
            "RS+FD[SUE-z]": self.audit_sue_z,
            "RS+FD[SUE-r]": self.audit_sue_r,
            "RS+FD[OUE-z]": self.audit_oue_z,
            "RS+FD[OUE-r]": self.audit_oue_r,
        }

        
    def get_params(self):
        """
        Get the parameters of the MultidimensionalLDPAuditor.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params()
        params['d'] = self.d
        params['lst_k'] = self.lst_k
        
        return params

    def set_params(self, **params):
        """
        Set the parameters of the MultidimensionalLDPAuditor.

        Parameters:
        ----------
        **params : dict
            Auditor parameters.

        Returns:
        -------
        self : auditor instance
            Auditor instance.
        """
        if 'd' in params:
            self.d = params.pop('d')
            self.lst_k = [self.k for _ in range(self.d)]
            self.v1 = [0 for _ in range(self.d)]
            self.v2 = [self.k-1 for _ in range(self.d)]
        if 'lst_k' in params:
            self.lst_k = params.pop('lst_k')
        return super().set_params(**params)

    #================================= Audit Methods for RS+FD Protocols =================================
    @ray.remote
    def audit_grr(self, random_state, trials, v, lst_k, d, epsilon, test_statistic):
        """
        Audits the Generalized Randomized Response (GRR) protocol for multidimensional data
        following the RS+FD solution.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : list
            True values for each attribute.
        lst_k : list
            List of domain sizes for each attribute.
        d : int
            Number of attributes (dimensions).
        epsilon : float
            Theoretical privacy budget.
        test_statistic : list
            Statistic to test against.

        Returns:
        -------
        int
            Count of successful inferences of the true value.
        """
        
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        setting_seed(random_state)
        np.random.seed(random_state)

        count = 0    
        for _ in range(trials):
            guess_att = np.random.randint(d) # Attacker has a random guess on the sampled attribute of the user
            ldp_val = RSpFD_GRR_Client(v, lst_k, d, epsilon)  
            count += ldp_val[guess_att] == test_statistic[guess_att] # We count each time the attacker guesses the attribute's value
        
        return count


    @ray.remote
    def audit_sue_z(self, random_state, trials, v, lst_k, d, epsilon, test_statistic):
        """
        Audits the Symmetric Unary Encoding with Zero-vector for fake data (SUE-z) protocol for multidimensional data
        with the RS+FD solution.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : list
            True values for each attribute.
        lst_k : list
            List of domain sizes for each attribute.
        d : int
            Number of attributes (dimensions).
        epsilon : float
            Theoretical privacy budget.
        test_statistic : list
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
            guess_att = np.random.randint(d)  # Attacker guesses the attribute
            ldp_val = RSpFD_UE_zero_Client(v, lst_k, d, epsilon, False)  
            count += attack_ue(ldp_val[guess_att], lst_k[guess_att]) == test_statistic[guess_att] # We count each time the attacker guesses the attribute's value
        
        return count

    @ray.remote
    def audit_sue_r(self, random_state, trials, v, lst_k, d, epsilon, test_statistic):
        """
        Audits the Symmetric Unary Encoding with Random-vector for fake data (SUE-r) protocol for multidimensional data
        with the RS+FD solution.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : list
            True values for each attribute.
        lst_k : list
            List of domain sizes for each attribute.
        d : int
            Number of attributes (dimensions).
        epsilon : float
            Theoretical privacy budget.
        test_statistic : list
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
            guess_att = np.random.randint(d)  # Attacker guesses the attribute
            ldp_val = RSpFD_UE_rnd_Client(v, lst_k, d, epsilon, False)  
            count += attack_ue(ldp_val[guess_att], lst_k[guess_att]) == test_statistic[guess_att] # We count each time the attacker guesses the attribute's value
        
        return count

    @ray.remote
    def audit_oue_z(self, random_state, trials, v, lst_k, d, epsilon, test_statistic):
        """
        Audits the Optimal Unary Encoding with Zero-vector for fake data (OUE-z) protocol for multidimensional data
        with the RS+FD solution.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : list
            True values for each attribute.
        lst_k : list
            List of domain sizes for each attribute.
        d : int
            Number of attributes (dimensions).
        epsilon : float
            Theoretical privacy budget.
        test_statistic : list
            Statistic to test against.
        optimal : bool
            Whether to use the optimal version of the protocol (default is True).

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
            guess_att = np.random.randint(d)  # Attacker guesses the attribute
            ldp_val = RSpFD_UE_zero_Client(v, lst_k, d, epsilon, True)  
            count += attack_ue(ldp_val[guess_att], lst_k[guess_att]) == test_statistic[guess_att] # We count each time the attacker guesses the attribute's value
        
        return count

    @ray.remote
    def audit_oue_r(self, random_state, trials, v, lst_k, d, epsilon, test_statistic):
        """
        Audits the Optimal Unary Encoding with Random-vector for fake data (OUE-r) protocol for multidimensional data
        with the RS+FD solution.

        Parameters:
        ----------
        random_state : int
            Random seed for reproducibility.
        trials : int
            Number of trials to run.
        v : list
            True values for each attribute.
        lst_k : list
            List of domain sizes for each attribute.
        d : int
            Number of attributes (dimensions).
        epsilon : float
            Theoretical privacy budget.
        test_statistic : list
            Statistic to test against.
        optimal : bool
            Whether to use the optimal version of the protocol (default is True).

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
            guess_att = np.random.randint(d)  # Attacker guesses the attribute
            ldp_val = RSpFD_UE_rnd_Client(v, lst_k, d, epsilon, True)  
            count += attack_ue(ldp_val[guess_att], lst_k[guess_att]) == test_statistic[guess_att] # We count each time the attacker guesses the attribute's value
        
        return count
    
    #================================= General Audit Method =================================
    def run_audit(self, protocol_name):
        """
        Runs the audit for the specified LDP protocol over multidimensional data.

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
            TP.append(protocol.remote(self, unique_seed, self.lst_trial_per_core[idx], self.v1, self.lst_k, self.d, self.epsilon, self.v1))
            FP.append(protocol.remote(self, unique_seed, self.lst_trial_per_core[idx], self.v2, self.lst_k, self.d, self.epsilon, self.v1))

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




