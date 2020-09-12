"""
Global Configuration for the project

@author: Angel Villar-Corrales
"""


CONFIG = {
    "paths": {
        "exp_path": "../experiments",
        "data_path": "../data",
        "results_path": "results"
    },
    "random_seed": 13,
    "num_workers": 4
}


DEFAULTS_CLASSIFICATION = {
    "data": {
        "batch_size": 128,      # number elemnts in batch
        "valid_size": 0.25,     # percentage of training data to use for validation
        "equalize": True        # If True, equalizing scat features along the same path
    },
    "classification": {
        "optimize_dims": True,  # if True, number of remove directions is optimized on val-set
        "min_dims": 50,         # first n_dims to consider for optimization
        "max_dims": 250,        # last n_dims to consider for optimization
        "step_dims": 20,        # step used for evaluating n_dims for optimization
        "num_dims": 120,        # number of dimensions to remove (if optimize_dims is False)
    }
}

DEFAULTS_CLUSTERING = {

}
