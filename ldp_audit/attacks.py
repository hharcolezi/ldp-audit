# general imports
import numpy as np
from numba import jit, prange
import xxhash

@jit(nopython=True)
def attack_ss(ss):
    """
    Privacy attack to Subset Selection (SS) protocol.

    Parameters:
    ----------
    ss : array
        Obfuscated subset of values.

    Returns:
    -------
    int
        A random inference of the true value.
    """
                
    return np.random.choice(ss)

@jit(nopython=True)
def attack_ue(ue_val, k):
    """
    Privacy attack to Unary Encoding (UE) protocols.

    Parameters:
    ----------
    ue_val : array
        Obfuscated vector.
    k : int
        Domain size.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    if np.sum(ue_val) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(np.where(ue_val == 1)[0])

@jit(nopython=True)
def attack_the(ue_val, k, thresh):
    """
    Privacy attack to Thresholding with Histogram Encoding (THE) protocol.

    Parameters:
    ----------
    ue_val : array
        Obfuscated vector.
    k : int
        Domain size.
    thresh : float
        Optimal threshold value.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    ss_the = np.where(ue_val > thresh)[0]
    if len(ss_the) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(ss_the)

@jit(nopython=True)
def attack_she(y, k, epsilon):
    """
    Privacy attack to Summation with Histogram Encoding (THE) protocol.

    Parameters:
    ----------
    y : array
        Obfuscated vector.
    k : int
        Domain size.
    epsilon : float
        Theoretical privacy guarantees.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    v_likelihood = np.zeros(k)
    for v in prange(k):
        x_v = np.zeros(k)
        x_v[v] = 1
        v_likelihood[v] = np.prod(np.exp(-np.abs(y - x_v) / (2/epsilon)))
    posterior_v = v_likelihood / np.sum(v_likelihood)
    m = max(posterior_v) 
    return np.random.choice(np.where(posterior_v == m)[0])

def attack_lh(val_seed, k, g):
    """
    Privacy attack to Local Hashing (LH) protocols.

    Parameters:
    ----------
    val_seed : tuple
        Obfuscated tuple (obfuscated value, seed as "hash function").
    k : int
        Domain size.
    g : int
        Hash domain size.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    lh_val = val_seed[0]
    rnd_seed = val_seed[1]

    ss_lh = []
    for v in range(k):
        if lh_val == (xxhash.xxh32(str(v), seed=rnd_seed).intdigest() % g):
            ss_lh.append(v)

    if len(ss_lh) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(ss_lh)

@jit(nopython=True)
def attack_gm(y, k, sigma):
    """
    Privacy attack to Gaussian Mechanisms (GM).

    Parameters:
    ----------
    y : array
        Obfuscated vector.
    k : int
        Domain size.
    sigma : float
        Sigma used for drawing noise.

    Returns:
    -------
    int
        A random inference of the true value.
    """

    # Prior probability for each v is uniform
    prior_v = 1 / k

    # Compute the likelihood P_Y(y|v) for each possible v
    v_likelihood = np.zeros(k)
    for v in prange(k):
        v_encoded = np.zeros(k)
        v_encoded[v] = 1.0  # One-hot encoding

        # Compute the L2 squared distance
        l2_squared = np.sum((y - v_encoded) ** 2)

        # Compute the likelihood using the Gaussian probability density function
        v_likelihood[v] = np.exp(-l2_squared / (2 * sigma**2))

    # Normalize the likelihood by multiplying by the prior and summing across all v
    posterior_v = v_likelihood * prior_v
    posterior_v /= np.sum(posterior_v)  # Normalization

    # Select the v with the highest posterior probability (randomized "argmax")
    m = np.max(posterior_v)
    return np.random.choice(np.where(posterior_v == m)[0])