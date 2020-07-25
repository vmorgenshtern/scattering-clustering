"""
Implementation of sparse subspace clustering via matching pursuits.
Adapted from:
    Noisy subspace clustering via matching pursuits
    M. Tschannen and H. Bölcskei

@author: Angel Villar-Corrales
"""

from tqdm import tqdm

from numba import jit
import numpy as np

import lib.clustering.spectral_clustering as spectral_clustering


def SSS_MP(X, tau, s_max, p_max=None, L=0, cluster=True, optimized=True, verbose=0):
    """
    Implementation of Sparse Subspace Clustering via Matching Pursuit
    Adapted from:
        Noisy subspace clustering via matching pursuits
        M. Tschannen and H. Bölcskei

    Args:
    -----
    X: numpy array
        (m,N) matrix containing the data points to cluster
    tau: float
        threshold for the representation error. Used as stop condition in MP
    s_max: integer
        maximum number of matching pursuit operations
    p_max: integer
        sparsity level of the representation
    L: integer
        estimated number of clusters
    cluster: boolean
        if True, spectral clustering is computed
    optimized: boolea
        if True, SSC-MP uses the numba-optimized Matching Pursuit algorithm
    """

    dim, N = X.shape[0], X.shape[1]
    C = np.zeros((N,N))
    labels = [None]*N
    if(p_max is None):
        p_max = N

    # for each vector in X, we find a representation based on the other vectors
    if(optimized):
        C = MP_iters(X, N, tau, s_max, p_max, verbose)
    else:
        for i in tqdm(range(N)):
            r = X[:, i]
            Xc = np.copy(X)
            Xc[:, i] = np.zeros(Xc[:, i].shape)
            coeffs = np.zeros(N)

            # matching pursuit
            num_iters = 0
            while( (np.linalg.norm(r) > tau) and (num_iters < s_max)
                    and (len(np.where(coeffs != 0)[0]) < p_max) ):
                idx = np.argmax(np.abs(np.matmul(Xc.T, r)))  # argmax operator (equation 3)
                coeff_idx = np.dot(Xc[:, idx].T, r) / np.square(np.linalg.norm(Xc[:, idx]))
                coeffs[idx] = coeffs[idx] + coeff_idx  # updating coefficient vector (equation 4)
                r = r - np.dot(Xc[:, idx].T, coeff_idx)  # updating residual (equation 5)
                num_iters = num_iters + 1
            C[:, i] = coeffs

    if(cluster):
        labels = spectral_clustering.spectral_clustering(C, L)

    return labels, C


@jit(nopython=True)
def MP_iters(X, N, tau, s_max, p_max, verbose):
    """
    Computing Matching Pursuit on a matrix having the datapoints as columns
    Optimized method using Numba compiler: out of all versions of the method, this one
    yielded the best running time
    """

    C = np.zeros((N,N))
    percs = []

    # for each vector in X, we find a representation based on the other vectors
    for i in range(N):
        perc = 100 * i // N
        if( perc % 10 == 0 and perc not in percs):
            percs.append(perc)
            if(verbose > 0):
                print(perc, "% of the vectors processed...")
        r = X[:, i]
        Xc = np.copy(X)
        Xc[:, i] = np.zeros(Xc[:, i].shape)
        coeffs = np.zeros(N)

        # matching pursuit
        num_iters = 0
        while( (np.linalg.norm(r) > tau) and (num_iters < s_max)
                and (len(np.where(coeffs != 0)[0]) < p_max) ):
            idx = np.argmax(np.abs(np.dot(Xc.T, r)))  # argmax operator (equation 3)
            coeff_idx = np.dot(Xc[:, idx].T, r) / np.square(np.linalg.norm(Xc[:, idx]))
            coeffs[idx] = coeffs[idx] + coeff_idx  # updating coefficient vector (equation 4)
            r = r - Xc[:, idx].T * coeff_idx  # updating residual (equation 5)
            num_iters = num_iters + 1

        C[:, i] = coeffs

    return C


#
