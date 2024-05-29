from scipy.special import erf
import numpy as np
from numba import jit, prange
from sys import maxsize
import xxhash

# [1] https://github.com/BorjaBalle/analytic-gaussian-mechanism/blob/master/agm-example.py
# [2] Balle & Wang. "Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising".
# [3] Dwork & Roth. "The algorithmic foundations of differential privacy".
# [4] Wang et al. "Local differential privacy for data collection and analysis".

def find_scale(epsilon, delta, Delta_2, tol=1.e-12):
    """
    Find the scale for the Analytical Gaussian Mechanism (AGM) [2].

    Parameters:
    ----------
    epsilon : float
        Theoretical privacy budget.
    delta : float
        Privacy parameter for approximate LDP.
    Delta_2 : float
        L2 sensitivity.
    tol : float
        Tolerance for numerical optimization (default is 1.e-12).

    Returns:
    -------
    float
        Calculated scale for the AGM.
    """
    
    # Code from [1] based on [2].
    def Phi(t):
        return 0.5 * (1.0 + erf(float(t) / np.sqrt(2.0)))

    def caseA(epsilon, s):
        return Phi(np.sqrt(epsilon * s)) - np.exp(epsilon) * Phi(-np.sqrt(epsilon * (s + 2.0)))

    def caseB(epsilon, s):
        return Phi(-np.sqrt(epsilon * s)) - np.exp(epsilon) * Phi(-np.sqrt(epsilon * (s + 2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if delta == delta_thr:
        alpha = 1.0
    else:
        if delta > delta_thr:
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: np.sqrt(1.0 + s / 2.0) - np.sqrt(s / 2.0)
        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: np.sqrt(1.0 + s / 2.0) + np.sqrt(s / 2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    return alpha * Delta_2 / np.sqrt(2 * epsilon)
    
@jit(nopython=True)
def GM_Client(input_data, k, sigma):
    """
    Gaussian Mechanism (GM) [3] for generating noisy data.

    Parameters:
    ----------
    input_data : int
        True value.
    k : int
        Domain size.
    sigma : float
        Standard deviation of the Gaussian noise.

    Returns:
    -------
    np.ndarray
        Noisy vector of length k.
    """

    
    input_ue_data = np.zeros(k)
    input_ue_data[input_data] = 1.0

    return input_ue_data + np.random.normal(loc=0, scale=sigma, size=k)
    
@jit(nopython=True)
def AGRR_Client(input_data, k, epsilon, delta):
    """
    Approximate Generalized Randomized Response (AGRR) [4]
    
    Parameters:
    ----------
    input_data : int
        True value.
    k : int
        Domain size.
    epsilon : float
        Theoretical privacy budget.
    delta : float
        Privacy parameter for approximate LDP.

    Returns:
    -------
    int
        Obfuscated value.
    """

    # AGRR parameters
    p = (np.exp(epsilon) + (k - 1) * delta) / (np.exp(epsilon) + k - 1)

    # Mapping domain size k to the range [0, ..., k-1]
    domain = np.arange(k)

    # GRR perturbation function
    if np.random.binomial(1, p) == 1:
        return input_data
    else:
        return np.random.choice(domain[domain != input_data])


@jit(nopython=True)
def ASUE_Client(input_data, k, epsilon, delta):
    """
    Approximate Symmetric Unary Encoding (ASUE) [4]
    
    Parameters:
    ----------
    input_data : int
        True value.
    k : int
        Domain size.
    epsilon : float
        Theoretical privacy budget.
    delta : float
        Privacy parameter for approximate LDP.

    Returns:
    -------
    np.ndarray
        Obfuscated unary encoded vector.
    """

    # Symmetric parameters (p+q = 1)
    p = (np.exp(epsilon) - np.sqrt(np.exp(epsilon) * (1 - delta) + delta)) / (np.exp(epsilon) - 1)
    q = (np.sqrt(np.exp(epsilon) * (1 - delta) + delta) - 1) / (np.exp(epsilon) - 1)

    # Unary encoding
    input_ue_data = np.zeros(k)
    if input_data is not None:
        input_ue_data[input_data] = 1

    # Initializing a zero-vector
    sanitized_vec = np.zeros(k)

    # UE perturbation function
    for ind in prange(k):
        rnd = np.random.random()
        if input_ue_data[ind] != 1:
            if rnd <= q:
                sanitized_vec[ind] = 1
        else:
            if rnd <= p:
                sanitized_vec[ind] = 1
    return sanitized_vec

def ALH_Client(input_data, epsilon, delta, use_optimal=True):
    """
    Approximate Local Hashing (ALH) [4].

    Parameters:
    ----------
    input_data : int
        True value.
    epsilon : float
        Theoretical privacy budget.
    delta : float
        Privacy parameter for approximate LDP.
    use_optimal : bool
        Flag to use optimal parameters for Approximate Optimal LH (AOLH) or Approximate Binary LH (ABLH).
    
    Returns:
    -------
    tuple
        Obfuscated value and random seed.
    """

    if use_optimal:  # Approximate OLH (AOLH) [4]
        g = max(2, int(np.round((-3 * np.exp(epsilon) * delta - np.sqrt(np.exp(epsilon) - 1) *
                                 np.sqrt((1 - delta) * (np.exp(epsilon) + delta - 9 * np.exp(epsilon) * delta - 1))
                                 + np.exp(epsilon) + 3 * delta - 1) / (2 * delta))))
    else:  # Approximate BLH (ABLH)
        g = 2

    # Generate random seed and hash the user's value
    rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
    hashed_input_data = xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g

    # LH perturbation function (i.e., GRR-based)
    sanitized_value = AGRR_Client(hashed_input_data, g, epsilon, delta)

    return sanitized_value, rnd_seed