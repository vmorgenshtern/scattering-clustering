"""
Methods for reading and processing command line arguments

 @author: Angel Villar-Corrales
"""

import os
import copy
import argparse

from CONFIG import DEFAULTS as def_params



def process_clustering_arguments():
    """
    Defining and processing command line arguments for clustering experiments
    """

    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--dataset_name', help="Name of the dataset to classify or " \
                        "cluster: ['mnist', 'fashion_mnist', 'svhn', 'usps', " \
                        "'mnist-test', 'coil-100', 'cifar']",
                        default="mnist")
    parser.add_argument('--verbose', help='verbosity level', type=int, default=0)
    parser.add_argument('--random_seed', help='Random Seed', type=int, default=0)
    parser.add_argument('--fname', help='Name of the file where results are stored')


    # data parameters
    parser.add_argument('--scattering', help="If True, clustering is performend on " \
                        "scattering feature reperesentation. Otherwise on images")
    parser.add_argument('--batch_size', help="Number of elements in batch for " \
                        "data loaders", type=int)
    parser.add_argument('--valid_size', help="Percentage of the training set to use " \
                        "for validation. In range [0,1]", type=float)
    parser.add_argument('--equalize', help="If True, scattering features are equalized so that " \
                        "features for the same path are on the same scale.")

    # dimensionality reduction
    parser.add_argument('--pca', help="Binary flag. If True, PCA is applied to rediced the " \
                        "dimensionality of scattering features prior to clustering")
    parser.add_argument('--pca_dims', help="Number of directions of variance to keep " \
                        "with PCA", type=int)

    # scattering
    parser.add_argument('--J', help="Support of the scattering features", type=int)
    parser.add_argument('--max_order', help="Max number of cascaded wavelets and modulus", type=int)
    parser.add_argument('--L', help="Number of rotations to evaluate", type=int)
    parser.add_argument('--shape', help="Shape (S,S) of the input features to the " \
                        "scattering network", type=int)

    # clustering parameters
    parser.add_argument('--uspec', help="If True, U-SPEC algorithm is used for clustering. " \
                        "Otherwise K-Means performs the clustering")
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

    args = parser.parse_args()

    # enforcing correct values
    dataset_name = args.dataset_name
    verbose = args.verbose
    random_seed = args.random_seed
    fname = args.fname
    assert dataset_name in ["mnist", "fashion_mnist", 'svhn', 'usps', 'mnist-test', \
                            'coil-100', 'cifar'], \
        f"ERROR! wrong 'dataset_name' parameter: {dataset_name}.\n Only ['mnist',"\
        f"'fashion_mnist', 'usps', 'mnist-test', 'cifar'] are allowed"
    args.equalize = (args.equalize == "True") if args.equalize != None else None
    args.pca = (args.pca == "True") if args.pca != None else None
    args.uspec = (args.uspec == "True") if args.uspec != None else None
    args.poc_preprocessing = (args.poc_preprocessing == "True") if args.poc_preprocessing != None else None
    args.scattering = (args.scattering == "True") if args.scattering != None else None

    # matching default values
    # params = copy.deepcopy(def_params)
    params = {}
    cl_params = vars(args)
    params["fname"] = fname
    for p in def_params["data"]:
        params[p] = cl_params[p] if(cl_params[p] is not None) else def_params["data"][p]
    for p in def_params["scattering"]:
        params[p] = cl_params[p] if(cl_params[p] is not None) else def_params["scattering"][p]
    for p in def_params["clustering"]:
        params[p] = cl_params[p] if(cl_params[p] is not None) else def_params["clustering"][p]
    params = argparse.Namespace(**params)

    return dataset_name, verbose, random_seed, params


def process_baseline_arguments():
    """
    Processing command line arguments for baseline clustering methods
    """

    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--dataset_name', help="Name of the dataset to classify or " \
                        "cluster: ['mnist', 'fashion_mnist', 'svhn', 'usps', " \
                        "'mnist-test', 'coil-100', 'cifar']",
                        default="mnist")
    parser.add_argument('--method', help="baseline clustering method: ['k_means', " \
                        "'spectral_clustering', 'dbscan']", default="k_means")
    parser.add_argument('--verbose', help='verbosity level', type=int, default=0)
    parser.add_argument('--random_seed', help='Random Seed', type=int, default=0)

    args = parser.parse_args()

    # enforcing correct values
    dataset_name = args.dataset_name
    method = args.method
    verbose = args.verbose
    random_seed = args.random_seed
    assert dataset_name in ["mnist", "fashion_mnist", 'svhn', 'usps', 'mnist-test', \
                            'coil-100', 'cifar'], \
        f"ERROR! wrong 'dataset_name' parameter: {dataset_name}.\n Only ['mnist',"\
        f"'fashion_mnist', 'usps', 'mnist-test', 'cifar'] are allowed"
    assert method in ["k_means", "spectral_clustering", "dbscan"]

    return dataset_name, method, verbose, random_seed



# DEPRECATED
def process_classification_arguments():
    """
    Defining and processing command line arguments for classification experiments
    """

    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--dataset_name', help="Name of the dataset to classify or " \
                        "cluster: ['mnist', 'fashion_mnist', 'svhn', 'usps', 'mnist-test']",
                        default="mnist")
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
    assert dataset_name in ["mnist", "fashion_mnist", 'svhn', 'usps', 'mnist-test'], \
        f"ERROR! wrong 'dataset_name' parameter: {dataset_name}.\n Only ['mnist',"\
        f"'fashion_mnist', 'usps', 'mnist-test'] are allowed"
    args.equalize = (args.equalize == "True") if args.equalize != None else None
    args.optimize_dims = (args.optimize_dims == "True") if args.optimize_dims != None else None

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


    #
