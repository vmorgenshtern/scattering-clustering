"""
Class for computing the Projection onto Orthogonal Complement (POC) in a skelear.PCA fashion
"""

import os

import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

from lib.projections.projection_orthogonal_complement import compute_prototype
from lib.dimensionality_reduction.dimensionality_reduction import compute_eigendecomposition


class POC():
    """
    Computing the Projection onto Orthogonal Complement (POC) algorithm.
    This class is structured to mimic the PCA from Sklearn
    """

    def __init__(self, n_dims=100, standarize=False):
        """
        Initializer of the POC object
        """

        self.standarize = standarize
        self.n_dims = n_dims

        self.eigenvectors = None
        self.prototype = None

        return


    def fit(self, data):
        """
        Fitting the POC object with data. Computing prototype and eigendecomposition
        """

        eigenvalues,\
            eigenvectors = compute_eigendecomposition(data=data, standarize=self.standarize)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        # computing prototype
        prototype = compute_prototype(data)

        return


    def fit_preprocessed(self, eigenvectors, prototype):
        """
        Fitting the POC object with an already computed and preprocessed prototype
        and eigenvectors
        """

        prototype = prototype[np.newaxis,:]
        self.prototype = prototype
        self.eigenvectors = eigenvectors

        return


    def transform(self, data, n_dims=None):
        """
        Transforming the data by projecting it onto the orthogonal complement of the
        directions of largest variance
        """

        if(n_dims is not None):
            self.n_dims = n_dims
        if(len(data.shape)>2):
            data = data.reshape(data.shape[0], -1)

        # shifting pointcloud to origin
        # shifted_data = data - self.prototype
        shifted_data = data

        # removing the eigenvectors associated to the largest eigenvalues
        eigenvectors_reduced = self.eigenvectors[:, self.n_dims:]

        # projecting onto low-dim space
        projected_data = np.matmul(eigenvectors_reduced.T, shifted_data.T).T

        return projected_data


    def fit_transform(self, data, n_dims=None):
        """
        Sequentially fitting the model and projecting the given data
        """

        self.fit(data)
        projected_data = self.transform(data, n_dims)

        return projected_data

#
