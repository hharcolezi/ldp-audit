# General imports
import warnings; warnings.simplefilter('ignore')
from numba.core.errors import NumbaExperimentalFeatureWarning
import numpy as np
import ray
from statsmodels.stats.proportion import proportion_confint
import xxhash

# Our imports
from .utils import setting_seed, LHO_Client
from .attacks import attack_lh
from .base_auditor import LDPAuditor

class LHOAuditor(LDPAuditor):
    """
    The LHOAuditor class is designed to audit the Local Hashing Only (LHO) protocol.

    Methods:
    -------
    run_audit(protocol_name: str) -> float
        Runs the audit for the LHO protocol and returns the estimated empirical accuracy.
    """

    def __init__(self, nb_trials: int = int(1e6), alpha: float = 1e-2, k: int = 2,
                 random_state: int = 42, n_jobs: int = -1, g: int = 2):
        """
        Initializes the LHOAuditor with the specified parameters.

        Parameters:
        ----------
        nb_trials : int
            The number of trials for the audit (default is 1e6).
        alpha : float
            The significance level for the Clopper-Pearson confidence intervals (default is 1e-2).
        k : int
            The domain size (default is 2).
        random_state : int
            The random seed for reproducibility (default is 42).
        n_jobs : int
            The number of CPU cores to use for parallel processing (-1 uses all available cores, default is None).
        g : int
            The hash domain size.
        """
        # Some non-used values for epsilon and delta to conform to the expected range of LDPAuditor.
        epsilon = 0.0
        delta = 0.0

        super().__init__(nb_trials, alpha, epsilon, delta, k, random_state, n_jobs)
        self.g = g

        # The protocol to audit
        self.protocols = {
            "LHO": self.audit_lho
        }

    def get_params(self):
        """
        Get the parameters of the LHOAuditor.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params()
        params['g'] = self.g
        return params

    def set_params(self, **params):
        """
        Set the parameters of the LHOAuditor.

        Parameters:
        ----------
        **params : dict
            Auditor parameters.

        Returns:
        -------
        self : auditor instance
            Auditor instance.
        """
        if 'g' in params:
            self.g = params.pop('g')
        return super().set_params(**params)

    @ray.remote
    def audit_lho(self, random_state, trials, v, k, g, test_statistic):
        """
        Audits the Local Hashing Only (LHO) protocol.

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
        g : int
            Hash domain size.
        test_statistic : int
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
            count += attack_lh(LHO_Client(v, g), k, g) == test_statistic

        return count

    def run_audit(self, protocol_name="LHO"):
        """
        Runs the audit for the LHO protocol.

        Parameters:
        ----------
        protocol_name : str
            The name of the LDP protocol to audit. Default is "LHO".

        Returns:
        -------
        float
            The estimated empirical accuracy.

        Raises:
        ------
        ValueError
            If the specified protocol name is not supported.
        """

        if protocol_name != "LHO":
            raise ValueError(f"Unsupported protocol: {protocol_name}")

        TP, FP = [], []  # initialize list for parallelized results
        for idx in range(self.nb_cores):
            unique_seed = xxhash.xxh32(protocol_name).intdigest() + self.random_state + idx
            TP.append(self.audit_lho.remote(self, unique_seed, self.lst_trial_per_core[idx], self.v1, self.k, self.g, self.v1))
            FP.append(self.audit_lho.remote(self, unique_seed, self.lst_trial_per_core[idx], self.v2, self.k, self.g, self.v1))

        # take results with get function from Ray library
        TP = sum(ray.get(TP))
        FP = sum(ray.get(FP))

        # Clopper-Pearson confidence intervals
        TPR = proportion_confint(TP, self.nb_trials, self.alpha / 2, 'beta')[0]
        FPR = proportion_confint(FP, self.nb_trials, self.alpha / 2, 'beta')[1]

        # empirical epsilon estimation 
        self.eps_emp = np.log(TPR / FPR)

        return self.eps_emp
