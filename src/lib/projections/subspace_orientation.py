"""
Methods, algorithms and auxiliary functions for computing the orientation between
subspaces and directions of variance

Scattering_Space/lib/projections
@author: Angel Villar-Corrales
"""

import os

import numpy as np
import scipy as sc



def check_colinearity_eigenvectors(eigenvectors, num_eigenvectors, verbose=0):
    """
    Checking how colinear eigenvectors from different classes are

    Args:
    -----
    eigenvectors: list
        list containing the arrays with the eigenvectors for each class
        [ [eigenvectors class 1], [eigenvectors class 2], ..., [eigenvectors class N]  ]
    num_eigenvectors: integer
        number of eigenvectors to compute
    verbose: integer
        verbosity level

    Returns:
    --------
    correlations: 4-dimensional numpy array
        4-dimensional tensor containing all pairwise correlations
        dimensions: [class of eigenvenctors, class of eigenvenctors, num of eigenvector, num of eigenvector]
        examples:
            the correlation between 1st and 2nd eigenvectors of classes 1 and 2 respectively will be: correlations[0,1,0,1]
            the correlation between 3rd and 1st eigenvectors of classes 3 and 2 respectively will be: correlations[2,1,2,0]
    """

    n = len(eigenvectors)
    correlations = np.empty((n,n,num_eigenvectors,num_eigenvectors))

    eigenvectors_matrices = [eigenvectors[i][:,:num_eigenvectors] for i in range(n)]

    for i in range(n):
        for j in range(n):
            cur_corr = np.dot(eigenvectors_matrices[i].T, eigenvectors_matrices[j])
            correlations[i,j,:,:] = cur_corr

    return correlations


def retrieve_principal_angles(angles):
    """
    Recursively retrieving the principal angles from the inner product matrix

    Args:
    -----
    angles: numpy array
        matrix to obtain the smallest angle from

    Returns:
    --------
    _ : list
        List with recursively computed principal angles. None, if the angle matrix is empty
    """

    if(angles.shape[0]<1):
        return [None]

    idx = np.unravel_index(np.argmin(angles, axis=None), angles.shape)
    new_principal_angle = angles[idx]
    angles = np.delete(angles, idx[0], axis=0)
    angles = np.delete(angles, idx[1], axis=1)

    next_angle = retrieve_principal_angles(angles)

    return [new_principal_angle] + next_angle


def compute_principal_angles(P, Q, n_dims=None):
    """
    Computing the principal angles between two subspaces

    Args:
    -----
    P, Q: numpy array
        matrices with the eigenvectors (as columns) which define the subspaces
        to compute the orientations.
    n_dims: integer
        number of largest vectors to consider when computing the principal angles.
        Since few directions dominate variance, we want to avoid correlating directions
        of large variance with directions of insignifiant variability

    Returns:
    --------
    principal_angles: numpy array
        list with the sorted principal angles between the subspaces
    """

    assert(P.shape==Q.shape)

    if(n_dims is not None and n_dims<P.shape[1]):
        P = P[:,:n_dims]
        Q = Q[:,:n_dims]

    M = np.matmul(P.T, Q)
    angles = np.arccos(np.abs(M))
    principal_angles = retrieve_principal_angles(angles)
    principal_angles = np.array(principal_angles)[:-1]

    return principal_angles



def compute_angle_statistics(angles):
    """
    Computing statistics of the principal angles between two subspaces

    Args:
    -----
    angles: numpy array
        list with the sorted principal angles between the subspaces
    """

    num_colinear = len(np.where(angles <= 0.1)[0])
    num_orthogonal = len(np.where(angles >= (3.14/2))[0])

    stats = {}
    stats["Total Angles"] = int(len(angles))
    stats["Num Colinear"] = int(num_colinear)
    stats["Num Orthogonal"] = int(num_orthogonal)
    stats["Max Angle"] = float(np.max(angles))
    stats["Min Angle"] = float(np.min(angles))
    stats["Mean Angle"] = float(np.mean(angles))
    stats["Median Angle"] = float(np.median(angles))

    return stats


def display_principal_angle_statistics(angle_stats, index=""):
    """
    Display principal angles statistics as a pandas dataframe

    Args:
    -----
    angle_stats: dictionary or list of dictionaries
        dictionary(ies) containing the precomputed angle statistics
    index: list
        names to assign to each row of the pandas dataframe
    """

    if type(angle_stats) is list:
        df = pd.DataFrame()
        for i, cur_stats in enumerate(angle_stats):
            df_ = pd.DataFrame.from_dict(cur_stats, orient="index", columns=[index[i]]).transpose()
            df_.rename(index={'0': index[i]}, inplace=True)
            df = pd.concat([df, df_])
    else:
        df = pd.DataFrame.from_dict(angle_stats, orient="index").transpose()
        df.rename(index={'0': index}, inplace=True)

    col_list = ["Total Angles", "Num Colinear", "Num Orthogonal",
                "Max Angle", "Min Angle", "Mean Angle", "Median Angle"]
    df = df[col_list]

    return df


def closeness_subspaces(bases1, bases2, n_dims=1, norm=True):
    """
    Measuring the closeness of two subspaces based on the trace
    and product of the bases

    Args:
    -----
    bases1, bases2: numpy array
        matrices containing the eigenvectors (basis) of the subspace as columns
    n_dims: integer
        number of dimensions to remove
    norm: boolean
        if True, normalizes the trace to the range [-1,1]
    """

    bases1, bases2 = bases1[:,:n_dims], bases2[:,:n_dims]
    first_product = np.matmul(bases1, bases1.T)
    second_product = np.matmul(first_product, bases2)
    third_product = np.matmul(second_product, bases2.T)
    closeness = np.trace(np.abs(third_product))
#     closeness = np.trace(third_product)

    if(norm):
        closeness = closeness/np.shape(bases1)[1]

    return closeness


def distace_closeness(points, eigenvectors_1, eigenvectors_2, n_dims=1):
    """
    Computing the closeness between two subspaces based on the distance between projections

    Args:
    -----
    points: numpy array
        samples to project onto both manifolds
    eigenvectors_1, eigenvectors_2: numpy array
        basis of the spaces
    """

    eigenvectors_1 = eigenvectors_1[:,:n_dims]
    eigenvectors_2 = eigenvectors_2[:,:n_dims]

    proj_matrix_1 = np.matmul(eigenvectors_1, eigenvectors_1.T)
    proj_matrix_2 = np.matmul(eigenvectors_2, eigenvectors_2.T)
    projections = np.matmul((proj_matrix_1 - proj_matrix_2), points.T)

    distances = np.square(np.linalg.norm(projections, axis=0))
    mean_distance = np.mean(distances)

    return mean_distance

#
