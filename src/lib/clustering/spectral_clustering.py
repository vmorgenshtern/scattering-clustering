"""
Custom implementation of normalized spectral clustering, closely following the paper
'Normalized Spectral Clustering' by Ng, Jordan, and Weiss (2002)

@author: Angel Villar-Corrales
"""

from tqdm import tqdm

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sklearn import cluster
import sklearn.metrics as metrics
import hnswlib


def spectral_clustering(C, L=None, affinity=True):
    """
    Computing Spectral Clustering on the affinity matrix C
        Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)

    Args:
    -----
    C: numpy array
        Affiniy-matrix/data-matrix to decompose and cluster
    L: integer
        Number of clusters to extract
    affinity: boolean
        If True, Computes the affinity graph from the input data matrix
    """

    #  computing affinity matrix from data-matrix
    if(affinity is False):
        C = compute_affinity_matrix(np.copy(C))

    # converting into affinity matrix
    C = np.abs(C) + np.abs(C).T
    N = C.shape[0]
    eps = 1e-15

    # computing normalized graph laplacian
    norm_matrix = np.diag(1/(np.sqrt(np.sum(C, axis=0)) + eps))
    LN = np.eye(N) - np.matmul(np.matmul(norm_matrix, C),  norm_matrix)

    # singular value decomposition (values in decreasing order)
    _, s, Vt = sp.linalg.svd(LN)
    V = Vt.T  # Vt is returned as a row-matrix. Therefore we transpose

    # estimation of L using eigengap heuristics
    if(L is None or L < 1):
        L, _ = eigengap_heuristics(LN)
        print(f"Eigengap detected {L} as the optimum number of clusters")

    # postprocessing
    VL = V[:, N-L:N]
    VL = VL / (np.linalg.norm(VL, axis=1, keepdims=True) + eps)

    # k-means
    kmeans = cluster.KMeans(n_clusters=L, random_state=13, max_iter=5000, n_jobs=-1).fit(VL)
    labels = kmeans.labels_

    return labels


def compute_affinity_matrix(data, method="knn", **kwargs):
    """
    Computing affinity matrix by applying kNN on the input data-matrix

    Args:
    -----
    C: numpy array
        Matrix with data-points in columns (dim, N)
    method: string
        methosd used to convert data to affinities ['knn', 'kernel']

    Returns:
    --------
    A: numpy array
        affinity graph containing
    """

    assert method in ['knn', 'kernel']

    if(method is "kernel"):
        affinity_matrix = kernel_affinity_matrix(data, **kwargs)

    elif(method is "knn"):
        affinity_matrix = neighbors_affinity_matrix(data, **kwargs)

    return affinity_matrix


def neighbors_affinity_matrix(data, k=5, **kwargs):
    """
    Computed an affinity matrix based on a kNN graph

    Args:
    -----
    data: numpy array
        data matrix containing all feature vectors (N, dims)
    k: integer
        number of neighbors used in the graph
    """

    num_elements, data_dim = data.shape

    # creating hns graph object and fitting features
    m = 8              # minimum number of outgoing edges for each node
    ef = 1000           # increase query "volume" to make results more stable
    neighbors_graph = hnswlib.Index(space='l2', dim=data_dim)
    neighbors_graph.init_index(max_elements=num_elements, ef_construction=ef, M=m)
    neighbors_graph.set_ef(ef)
    neighbors_graph.add_items(data, np.arange(num_elements))


    # retrieving kNN for each data-point and fitting affinity matrix
    affinity_matrix = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        idx, distance = neighbors_graph.knn_query(data[i,:], k=k+1)
        affinity_matrix[i, idx] = 1

     #enforcing zero diagonal, removing trivial retrieval
    np.fill_diagonal(affinity_matrix, 0)

    return affinity_matrix


def kernel_affinity_matrix(data, kernel_type="gaussian", beta=1):
    """
    Computed an affinity matrix based on evaluating a kernel on pairwise distances

    Args:
    -----
    data: numpy array
        data matrix containing all feature vectors (N, dims)
    kernel_type: string
        name of the kernel to use to evaluate distances ['gaussian']
    beta: integer
        multiplicative factor for the gaussian kernel
    """

    assert (kernel_type in ['gaussian'], "Only ['gaussian'] kernels are available")

    num_elements, data_dim = data.shape

    # selecting kernel
    if(kernel_type == "gaussian"):
        kernel = lambda distances : np.exp(-np.square(distances) * beta)

    # normalizing feature vectors so that distances are bounded
    data = data / np.linalg.norm(data, axis=1).reshape(-1,1)
    distances = metrics.pairwise_distances(data, data, metric="euclidean")

    affinity_matrix = kernel(distances)
    np.fill_diagonal(affinity_matrix, 0)

    return affinity_matrix


def eigengap_heuristics(LN, alpha=100, laplacian=True):
    """
    Implementation of the eigengap heuristics to find the optimum value of k

    Args:
    -----
    LN: numpy array
        Graph laplacian or adjacency matrix.
    alpha: integer
        Number of eigenvalues used to measure the eigengap
    laplacian: Boolean
        If True, LN corresponds to the graph laplacian. Otherwise it corresponds
        to the adjacency matrix and the laplacian is computed

    Returns:
    --------
    eigengap: integer
        estimated optimum number of clusters
    eigenvals: numpy array
        one diemsnional array containing the eigenvalues sorted ascendingly
    """

    if(laplacian == False):
        # converting into affinity matrix
        LN = np.abs(LN) + np.abs(LN).T
        N = LN.shape[0]
        eps = 1e-15
        # computing normalized graph laplacian
        norm_matrix = np.diag(1/(np.sqrt(np.sum(LN, axis=0)) + eps))
        LN = np.eye(N) - np.matmul(np.matmul(norm_matrix, LN),  norm_matrix)

    # eigenvalue decomposition and sorting
    eigenvals, _ = np.linalg.eig(LN)
    eigenvals = np.sort(eigenvals)[::]

    # computing largest eigengap
    if(alpha > 0):
        eigenvals = eigenvals[:alpha]
    diff_eigenvals = np.diff(eigenvals)[1:]
    eigengap = np.argmax(diff_eigenvals) + 2

    return eigengap, eigenvals


def display_eigengap(eigengap, eigenvals):
    """
    Displaying a plot with the sorted eigenvalues and highlighting the eigengap
    """

    plt.figure(figsize=(14,4))

    plt.subplot(1,2,1)
    x = np.arange(len(eigenvals))
    plt.scatter(x, eigenvals)
    plt.grid()
    plt.title("Sorted Eigenvalues")

    idx_min = max(0, eigengap - 10)
    idx_max = min(len(eigenvals), eigengap + 10)

    plt.subplot(1,2,2)
    x = np.arange(idx_min, idx_max)
    plt.scatter(x, eigenvals[idx_min:idx_max], label="Eigenvalues")
    plt.xticks(x[::4])
    plt.grid()
    plt.title("Zoom in Eigengap")
    offset = 0.01
    ymin = np.min(eigenvals[idx_min:idx_max]) - offset
    ymax = np.max(eigenvals[idx_min:idx_max]) + offset
    plt.vlines(eigengap-0.5, ymin=ymin, ymax=ymax, color="red", label=f"Eigengap={eigengap-1}")
    plt.ylim(ymin, ymax)
    plt.legend(loc="best")

    return


def cross_validation_spectral_clustering(data, original_labels, method="knn", candidates=None,
                                         verbose=1):
    """
    Applying cross validation to estimate the optimal hyperparameters of the
    spectral clustering algorithm

    Args:
    -----
    data: numpy array
        data samples used to construct the affinity matrix
    original_labels: numpy array
        annotated labels of the samples. Needed to compute score
    method: string
        method used to compute the affinity matrix ['knn', 'kernel']
    candidates: list
        list of candidate values to apply during the cross-validation procedure
    verbose: integer
        verbosity level

    Returns:
    --------
    opt_candidate: float
        number of neighbors or kernel width that yields the highest score
    opt_score: float
        highest score obtained for any of the candidates
    extra_data: dictionary
        dict containing all evaluated candidates, all clustering-scores measured
        and some cross-validation metadata
    """

    knns = np.arange(1,52,5)
    scores = []

    # initializing method for constructing the affinity matrix given parameters
    if(method is "knn"):
        construct_affinity = lambda k: neighbors_affinity_matrix(data, k=k)
        if(candidates is None):
            candidates = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    elif(method is "kernel"):
        construct_affinity = lambda b: kernel_affinity_matrix(data, beta=b)
        if(candidates is None):
            candidates = [128, 100, 64, 40, 32, 20, 16, 8, 4, 2, 1]

    if(verbose > 0):
        iterator = tqdm(candidates)
    else:
        iterator = candidates

    # computing clustering-score for each value in the candidates list
    for k in iterator:
        cur_affinity_matrix = construct_affinity(k=k)
        predicted_lbls = spectral_clustering(cur_affinity_matrix, L=10)
        cur_score = metrics.adjusted_rand_score(original_labels, predicted_lbls)
        scores.append(cur_score)

    # obtaining optimum candidate
    opt_idx = np.argmax(scores)
    opt_score = scores[opt_idx]
    opt_candidate = candidates[opt_idx]
    extra_data = {}
    extra_data["candidates"] = candidates
    extra_data["scores"] = scores
    extra_data["method"] = method

    if(verbose > 1):
        print(f"Optimum candidate value: {opt_candidate}")

    return opt_candidate, opt_score, extra_data


def display_cross_validation(scores, candidates, method, **kwargs):
    """
    Displaying a plot with the cluster score as a function of the evaluated
    hyperparameter values

    Args:
    -----
    scores, candidates: list
        lists containing the cluster-scores and the candidate values respectively
    method: string
        method used to construct the affinity graph ['kernel', 'knn']
    """

    if("ax" not in kwargs):
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(8,5)
    else:
        ax = kwargs["ax"]

    ax.grid()
    ax.plot(candidates, scores, linewidth=3)
    ax.scatter(candidates, scores, linewidth=3)

    ax.set_title("Validation Scores", fontsize=16)
    if(method is "knn"):
        ax.set_xlabel("Number of neighbors", fontsize=14)
    elif(method is "kernel"):
        ax.set_xlabel("Kernel width", fontsize=14)
    # plt.show()

    return


#
