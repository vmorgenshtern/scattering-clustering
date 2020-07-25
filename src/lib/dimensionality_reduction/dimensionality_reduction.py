"""
Methods for dimensionality reduction
"""

import os

import numpy as np
from sklearn import decomposition
from sklearn import manifold
from sklearn.preprocessing import StandardScaler


def pca(data, target_dimensions=3):
    """
    Using principal component analyisis for dimensionality reduction

    Args:
    -----
    data: numpy array [N,X]
        high dimensional data
    target_dimensions: Integer, default=3
        Number of componenets that we keep. Dimensionality of the output
    low_dim_data: numpy array [N, target_dimensions]
        low dimensional projection/mapping of the data
    """

    pca = decomposition.PCA(n_components=target_dimensions)
    pca.fit(data)
    low_dim_data = pca.transform(data)

    return low_dim_data


def tsne(data, target_dimensions=2):
    """
    Using t-distributed Stochastic Neighbor Embedding (t-sne) for dimensionality reduction

    Args:
    -----
    data: numpy array [N,X]
        high dimensional data
    target_dimensions: Integer, default=3
        Number of componenets that we keep. Dimensionality of the output
    low_dim_data: numpy array [N, target_dimensions]
        low dimensional projection/mapping of the data
    """

    tsne = manifold.TSNE(n_components=target_dimensions, verbose=1)
    low_dim_data = tsne.fit_transform(data)

    return low_dim_data


def compute_eigendecomposition(data, standarize=True):
    """
    Computing eigenvalues and eigenvectors given an array with data points

    Args:
    -----
    data: numpy array
        array with the input data

    Returns:
    --------
    eigenvalues: numpy array
        array with the eigenvalues of the covariance matrix sorted by magnitude in descending order
    eigenvectors: numpy array
        Matrix with the eigenvectors corresponding to the eigenvalues. Eigenvectors are aranged
        as columns of the matrix
    """

    # reshaping to feature vectors if necessary
    if(len(data.shape)>2):
        data = data.reshape(data.shape[0],-1)

    # standarizing vectors to zero-mean and unit-variance
    if(standarize):
        standardized_data = StandardScaler().fit_transform(data)
    else:
        standardized_data = data

    # computing covariance matrix of the data
    data_matrix = np.cov(standardized_data.T)

    # computing eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(data_matrix)

    # sorting by magnitude
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].astype(float)
    eigenvectors = eigenvectors[:,idx].astype(float)

    return eigenvalues, eigenvectors



#
