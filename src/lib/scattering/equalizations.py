"""
Methods for equalization of the scattering coefficients

@author: Angel Villar-Corrales
"""

import os
import json
import datetime

import numpy as np
import torch


def mean_variance_equalization(features, mean=None, std=None):
    """
    Computing the mean and variance of all features and standarizing to have
    zero-mean and unit variance

    Args:
    -----
    features: numpy array
        ndarray with shape (Batch,Path,Height,Width) containing the scattering features
    mean, std: numpy array
        precomputed means and variances used for equalization
    """

    if(mean is None or std is None):
        mean = features.mean(axis=0)
        std = features.std(axis=0)

    eps = 1e-13  # small epsilon for numerical stability
    features_eq = (features-mean)/(std+eps)

    return features_eq, mean, std


def max_norm_equalization(features, max_norms=None):
    """
    Equalizing the features by normalizing each coefficient by the norm of each
    path accross the complete set

    Args:
    -----
    features: numpy array
        ndarray with shape (Batch,Path,Height,Width) containing the scattering features
    max_norms: numpy array
        norms used for normalizing each path of the scattering network
    """

    if(len(features.shape)>4):
        features = features.reshape(features.shape[0], -1, *features.shape[-2:])

    if(max_norms is None):
        coeff_norms = np.linalg.norm(features, axis=(2,3))
        max_norms = np.max(coeff_norms, axis=0)

    features_eq = np.swapaxes(features, 3, 1) / max_norms
    features_eq = np.swapaxes(features_eq, 3, 1)

    return features_eq, max_norms


#
