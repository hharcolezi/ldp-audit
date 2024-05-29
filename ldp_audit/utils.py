# general imports
import numpy as np
from numba import jit
from scipy.optimize import minimize_scalar
from sys import maxsize
import xxhash

def LHO_Client(input_data, g):
    """
    Local Hashing Only (LHO) protocol, i.e., no LDP perturbation after hashing.

    Parameters:
    ----------
    input_data : int
        The user's true value to be hashed.
    g : int
        The hash domain size.

    Returns:
    -------
    tuple
        A tuple containing the hashed value and the random seed used for hashing.
    """

    # Generate random seed and hash the user's value
    rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
    hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)

    return (hashed_input_data, rnd_seed)


@jit(nopython=True)
def setting_seed(random_state):
    """ 
    Function to set seed for reproducibility.
    
    Calling numpy.random.seed() from interpreted code will 
    seed the NumPy random generator, not the Numba random generator.
    Check: https://numba.readthedocs.io/en/stable/reference/numpysupported.html

    Parameters:
    ----------
    random_state : int
        The random seed for reproducibility.
    """
    
    np.random.seed(random_state)

@jit(nopython=True)
def find_tresh(tresh, epsilon):    
    """
    Objective function for numerical optimization of thresh.

    Parameters:
    ----------
    tresh : float
        Threshold value for THE protocol.
    epsilon : float
        Privacy guarantee.

    Returns:
    -------
    float
        Variance (or MSE) when using a given epsilon/tresh with THE.
    """
    
    return (2 * (np.exp(epsilon*tresh/2)) - 1) / (1 + (np.exp(epsilon*(tresh-1/2))) - 2*(np.exp(epsilon*tresh/2)))**2
