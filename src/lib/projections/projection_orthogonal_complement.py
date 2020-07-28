"""
Methods for the 'Projection onto Orthogonal Complement algorithm'

@author: Angel Villar-Corrales
"""

import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

from lib.projections.dimensionality_reduction import compute_eigendecomposition


def extract_cluster_features(data, labels, cluster_id=0, prot_method="mean", standarize=False):
    """
    Computing the relevant features: prototype, eigenvectors and
    eigenvalues for a particular cluster

    Args:
    -----
    data: numpy array
        matrix containing the data points as columns (N, dims)
    labels: numpy array/list
        list with the label of each data point (N)
    cluster_id: integer
        number of the cluster we want to compute the statistics from
    prot_method: string
        statistic used to compute the class prototype ['mean', 'median']
    standarize: boolean
        if true, data feautures are standarized to be zero-mean and unit-variance

    Returns:
    --------
    prototype: numpy array
        class prototype. Corresponds to the mean/median of the class
    eigenvectors: numpy array
        eigenvector matrix of the class data-matrix, eigenvectors are sorted
        in descending order of spanned variance
    eigenvalues: numpy array
        eigenvalues from the data matrix sorted in descending order
    """

    # enforcing corrct shape and values for the parameters
    assert prot_method in ["mean", "median"]

    if(len(data.shape) > 2):
        data = data.reshape(data.shape[0], -1)
    if(cluster_id not in labels):
        print(f"There are no data points with label: {cluster_id}")
        return None, None

    # extracting features
    classwise_data = get_classwise_data(data=data, labels=labels, label=cluster_id)
    prototype = compute_prototype(data=classwise_data, method=prot_method)
    eigenvalues, eigenvectors = compute_eigendecomposition(data=classwise_data,
                                                           standarize=standarize)

    return classwise_data, prototype, eigenvectors, eigenvalues


def compute_prototype(data, method="mean"):
    """
    Computing the class prototype by taking the mean or median of the data

    Args:
    -----
    data: numpy array
        array-like object with the data
    prot_method: string
        statistic used to compute the class prototype ['mean', 'median']

    Returns:
    --------
    prototype: numpy array
        class prototype. Corresponds to the mean/median of the class
    """

    assert method in ["mean", "median"]

    if(prot_method == "mean"):
        prototype = np.mean(classwise_data, axis=0)
    elif(prot_method == "median"):
        prototype = np.median(classwise_data, axis=0)

    return


def get_classwise_data(data, labels, label=0, verbose=0):
    """
    Obtaining features and labels corresponding to a particular class

    Args:
    -----
    data: numpy array
        array-like object with the data
    labels: np array or list
        arraylike object with the labels corresponding to the data
    label: integer
        label corresponding to the data that we want to extract
    verbose: integer
        verbosity level

    Returns:
    --------
    classwise_data: np array
        array with the data corresponding to the desired class
    """

    idx = np.where(labels==label)[0]
    classwise_data = data[idx,:]
    if(verbose>0):
        print(f"There are {len(idx)} datapoints with label {label}")

    return classwise_data


#
