"""
Methods for computing subspace orientation and correltations through
the principal angles
"""

import numpy as np
import pandas as pd


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


def subspace_affinity(angles):
    """
    Computing the affinity between two spaces given the principal angles
    """

    angles = np.array(angles, dtype=float)
    corrs = np.sum( np.power( np.cos(angles),2) )
    affinity = np.sqrt(corrs / len(angles))

    return affinity


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
