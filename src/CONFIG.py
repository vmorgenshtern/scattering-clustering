"""
Global Configuration for the experiments and default parameters

@author: Angel Villar-Corrales
"""


# Global Configurations
CONFIG = {
    "paths": {
        "exp_path": "../experiments",
        "data_path": "../data",
        "results_path": "results",
        "plots_path": "plots"
    },
    "random_seed": 13,
    "num_workers": 4
}


# Defaul values for parameters
DEFAULTS = {
    "fname": "",                   # File where results are stored
    "data": {
        "scattering": True,        # If True, clustering is performed on scat. domain
        "batch_size": 128,         # number elemnts in batch
        "valid_size": 0.0,         # percentage of training data to use for validation
        "equalize": True,          # If True, equalizing scat features along the same path
        "pca": False,              # If True, dims. of scat features is reduced using PCA
        "pca_dims": 1000           # Number of eigenvectors to keep
    },
    "classification": {
        "optimize_dims": True,     # if True, number of remove directions is optimized on val-set
        "min_dims": 50,            # first n_dims to consider for optimization
        "max_dims": 250,           # last n_dims to consider for optimization
        "step_dims": 20,           # step used for evaluating n_dims for optimization
        "num_dims": 120,           # number of dimensions to remove (if optimize_dims is False)
    },
    "clustering": {
        "uspec": True,             # if True, USPE performs the clustering, otherwise k-means
        "poc_preprocessing": True, # if True, top directions of variance are removed using POC
        "num_dims": 2,             # number of directions to remove with POC
        "num_candidates": 10000,   # Number of candidates randomly rampled from DB for U-SPEC
        "num_reps": 1000,          # Number of candidates considered in U-SPEC
        "num_clusters": 10         # Targer number of clusters
    },
    "scattering": {
        "J": 3,                    # Scale of the wavelet filters (2^J, 2^J)
        "L": 8,                    # Number of rotations to consider in the Morlet wavelet
        "max_order": 2,            # Depth of the scattering layer
        "shape": 32                # shape (S,S) of the images input to the ScatNet
    }
}
