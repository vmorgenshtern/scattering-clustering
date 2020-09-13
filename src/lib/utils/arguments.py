"""
Methods for reading and processing command line arguments

 @author: Angel Villar-Corrales
"""

import os
import copy
import argparse

from CONFIG import DEFAULTS as def_params


def process_classification_arguments():
    """
    Defining and processing command line arguments for classification experiments
    """

    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--dataset_name', help="Name of the dataset to classify or " \
                        "cluster: ['mnist', 'fashion_mnist', 'svhn']", default="mnist")
    parser.add_argument('--verbose', help='verbosity level', type=int, default=0)


    # data parameters
    parser.add_argument('--batch_size', help="Number of elements in batch for " \
                        "data loaders", type=int)
    parser.add_argument('--equalize', help="If True, scattering features are equalized so that " \
                        "features for the same path are on the same scale.")

    # classification parameters
    parser.add_argument('--optimize_dims', help="If True, number of directions to remove is " \
                        "optimized using the validation set. [True/False]")
    parser.add_argument('--min_dims', help="First number of directios to use for " \
                        "parameter optimization", type=int)
    parser.add_argument('--max_dims', help="First number of directios to use for " \
                        "parameter optimization", type=int)
    parser.add_argument('--step_dims', help="Step used for evaluating n_dims for optimization ",
                        type=int)
    parser.add_argument('--num_dims', help="Number of dimensions to remove (overwritten if " \
                        "optimize_dims is False) ", type=int)

    # clustering parameters

    args = parser.parse_args()

    # enforcing correct values
    dataset_name = args.dataset_name
    verbose = args.verbose
    assert dataset_name in ["mnist", "fashion_mnist", 'svhn'], \
        f"ERROR! wrong 'dataset_name' parameter: {dataset_name}.\n Only ['mnist',"\
        f"'fashion_mnist'] are allowed"
    args.equalize = (args.equalize == True) if args.equalize != None else None
    args.optimize_dims = (args.optimize_dims == True) if args.optimize_dims != None else None

    # matching default values
    # params = copy.deepcopy(def_params)
    params = {}
    cl_params = vars(args)
    for p in def_params["data"]:
        params[p] = cl_params[p] if(cl_params[p] is not None) \
                                 else def_params["data"][p]
    for p in def_params["classification"]:
        params[p] = cl_params[p] if(cl_params[p] is not None) \
                                 else def_params["classification"][p]
    params = argparse.Namespace(**params)

    return dataset_name, verbose, params


def process_clustering_arguments():
    """
    Defining and processing command line arguments for clustering experiments
    """

    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--dataset_name', help="Name of the dataset to classify or " \
                        "cluster: ['mnist', 'fashion_mnist', 'svhn']", default="mnist")
    parser.add_argument('--verbose', help='verbosity level', type=int, default=0)

    # data parameters
    parser.add_argument('--batch_size', help="Number of elements in batch for " \
                        "data loaders", type=int)
    parser.add_argument('--valid_size', help="Percentage of the training set to use " \
                        "for validation. In range [0,1]", type=float)
    parser.add_argument('--equalize', help="If True, scattering features are equalized so that " \
                        "features for the same path are on the same scale.")

    # clustering parameters
    parser.add_argument('--num_candidates', help="Number of sample candidates for the sampling " \
                        "stage of the USPEC algorithm.", type=int)
    parser.add_argument('--num_reps', help="Number of sample representatives for the sampling " \
                        "stage of the USPEC algorithm.", type=int)
    parser.add_argument('--num_clusters', help="Number of clusters to extract from the data.",\
                        type=int)
    parser.add_argument('--poc_preprocessing', help="if True, the POC algorithm is a "\
                        "preprocessing step to the clustering algorithm")
    parser.add_argument('--num_dims', help="Number of directions to remove in the POC "\
                        "preprocessing step", type=int)


    # clustering parameters

    args = parser.parse_args()

    # enforcing correct values
    dataset_name = args.dataset_name
    verbose = args.verbose
    assert dataset_name in ["mnist", "fashion_mnist", 'svhn'], \
        f"ERROR! wrong 'dataset_name' parameter: {dataset_name}.\n Only ['mnist',"\
        f"'fashion_mnist'] are allowed"
    args.equalize = (args.equalize == True) if args.equalize != None else None
    args.poc_preprocessing = (args.poc_preprocessing == True) if args.poc_preprocessing != None else None

    # matching default values
    # params = copy.deepcopy(def_params)
    params = {}
    cl_params = vars(args)
    for p in def_params["data"]:
        params[p] = cl_params[p] if(cl_params[p] is not None) else def_params["data"][p]
    for p in def_params["clustering"]:
        params[p] = cl_params[p] if(cl_params[p] is not None) else def_params["clustering"][p]
    params = argparse.Namespace(**params)

    return dataset_name, verbose, params


    #
