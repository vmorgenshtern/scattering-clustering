"""
Methods for the 'Projection onto Orthogonal Complement algorithm'

@author: Angel Villar-Corrales
"""

from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

from lib.projections.dimensionality_reduction import compute_eigendecomposition


def projections_classifier(points, eigenvectors, prototypes, n_directions):
    """
    Classifying datapoints based on the POC algorithm

    Args:
    -----
    points: np array
        array with the points to project
    eigenvectors: list of np arrays
        list with the eigenvector matrix for each class that we want to project onto
    prototypes: list of np arrays
        list with the class-prototype for each class that we want to project onto
    n_directions: int
        number of directions to remove.

    Returns:
    --------
    labels: np array
        array containing the predicted labels for each datapoint
    min_distances: np array
        array containing the min distances
    """

    distances = []

    if(len(points.shape)>2):
        points = points.reshape(points.shape[0], -1)
    if(not isinstance(points, np.ndarray)):
        points = points.numpy()

    # removing the eigenvectors associated to the largest eigenvalues, projecting
    # the samples and computing the distance in the low-dimensional space
    for i in range(len(prototypes)):
        eigenvectors_reduced = eigenvectors[i][:, n_directions:]
        current_distances = points - prototypes[i]
        projected_distances = np.matmul(eigenvectors_reduced.T, current_distances.T).T
        projected_distances = np.linalg.norm(projected_distances, axis=1)
        distances.append(projected_distances)

    # chosing the labels corresponding to the smallest distances
    distances = np.array(distances)
    labels = np.argmin(distances, axis=0)
    min_distances = np.min(distances, axis=0)

    return labels, min_distances


def get_features_all_classes(data, labels, cluster_ids, verbose=0, **kwargs):
    """
    Extracting the feautures for all classes

    Args:
    -----
    data: numpy array
        matrix containing the data points as columns (N, dims)
    labels: numpy array/list
        list with the label of each data point (N)
    cluster_ids: list
        list with the labels to extract the features from

    Returns:
    --------
    class_data: list
        list with the data samples for each label
    prototypes: list
        list containing each class prototype
    eigenvectors: list
        list containing the eigenvectors for each label
    """

    class_data = []
    prototypes = []
    eigenvectors = []

    if(verbose==0):
        iterator = cluster_ids
    else:
        iterator = tqdm(cluster_ids)

    for id in iterator:
        cur_class_data, cur_prototype, cur_eigenvectors, \
            cur_eigenvalues = extract_cluster_features(data, labels, cluster_id=id, **kwargs)
        class_data.append(cur_class_data)
        prototypes.append(cur_prototype)
        eigenvectors.append(cur_eigenvectors)
        # eigenvalues.append(cur_eigenvalues)

    return class_data, prototypes, eigenvectors


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
    if(not isinstance(data, np.ndarray)):
        data = data.numpy()
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

    if(method == "mean"):
        prototype = np.mean(data, axis=0)
    elif(method == "median"):
        prototype = np.median(data, axis=0)

    return prototype


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
