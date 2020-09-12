"""
Performing image classification by applying the POC algorithm on scattering
representations of the original small images

@author: Angel Villar-Corrales
"""

import os
import json

import numpy as np
import torch

from lib.utils.arguments import process_classification_arguments
from lib.utils.utils import create_directory
from lib.data.data_loading import ClassificationDataset
from lib.data.data_processing import convert_loader_to_scat
from lib.projections.projection_orthogonal_complement import get_features_all_classes, \
    extract_cluster_features, projections_classifier, optimize_dimensionality
from lib.scattering.scattering_methods import scattering_layer
from CONFIG import CONFIG


def classification_experiment(dataset_name, params, verbose):
    """
    Performing a classification experiment using the POC algorithm on scattering
    features of the original images
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading data and breaking it into different dataset splits
    dataset = ClassificationDataset(data_path=CONFIG["paths"]["data_path"],
                                    dataset_name=dataset_name,
                                    valid_size=params.valid_size)
    train_loader = dataset.get_data_loader(split="train", batch_size=params.batch_size)
    valid_loader = dataset.get_data_loader(split="valid", batch_size=params.batch_size)
    test_loader = dataset.get_data_loader(split="test", batch_size=params.batch_size)

    # computing scattering representations
    scattering_net, _ = scattering_layer()
    scattering_net = scattering_net.cuda() if device.type == 'cuda' else scattering_net
    if verbose > 0: print("Computing scattering features for training set...")
    train_imgs, train_scat_features, train_labels = \
        convert_loader_to_scat(train_loader, scattering=scattering_net,
                               device=device, equalize=params.equalize,
                               verbose=verbose)
    if verbose > 0: print("Computing scattering features for validation set...")
    valid_imgs, valid_scat_features, valid_labels = \
        convert_loader_to_scat(valid_loader, scattering=scattering_net,
                               device=device, equalize=params.equalize,
                               verbose=verbose)
    if verbose > 0: print("Computing scattering features for test set...")
    test_imgs, test_scat_features, test_labels = \
        convert_loader_to_scat(test_loader, scattering=scattering_net,
                               device=device, equalize=params.equalize,
                               verbose=verbose)
    n_labels = len(np.unique(train_labels))

    # feature extraction => class eigenvectors and class prototypes
    if verbose > 0: print("Extracting class eigenvetors and prototypes...")
    cluster_ids = np.arange(n_labels).tolist()
    classwise_data, prototypes, eigenvectors = \
        get_features_all_classes(data=train_scat_features, labels=train_labels, verbose=verbose,
                                 cluster_ids=cluster_ids, standarize=False)

    # optimizing number of directions on validation set
    if(params.optimize_dims):
        if verbose > 0: print("Optimizing number of directions to remove...")
        direction_candidates = np.arange(params.min_dims, params.max_dims, params.step_dims)
        num_dims, max_acc, _ =  optimize_dimensionality(data=valid_scat_features,
                                                        labels=valid_labels,
                                                        dims=direction_candidates,
                                                        prototypes=prototypes,
                                                        eigenvectors=eigenvectors,
                                                        verbose=verbose)
    else:
        num_dims = params.num_dims

    # evaluating on test set
    if verbose > 0: print("Evaluating test set...")
    pred_test_labels_scat, _ = projections_classifier(points=test_scat_features,
                                                      prototypes=prototypes,
                                                      eigenvectors=eigenvectors,
                                                      n_directions=num_dims)

    n_correct_labels_scat = len(np.where(pred_test_labels_scat == test_labels)[0])
    test_set_acc_scat = 100 * n_correct_labels_scat / len(test_labels)

    print(f"Test set accuracy results:")
    print(f"    {round(test_set_acc_scat, 3)}%")

    # loading previous results, if any
    results_path = os.path.join(os.getcwd(), CONFIG["paths"]["results_path"])
    _ = create_directory(results_path)
    results_file = os.path.join(results_path, "classification_results.json")
    if(os.path.exists(results_file)):
        with open(results_file) as f:
            data = json.load(f)
            n_exps = len(list(data.keys()))
    else:
        data = {}
        n_exps = 0
    # saving experiment parameters and results
    with open(results_file, "w") as f:
        cur_exp = {}
        cur_exp["params"] = {}
        cur_exp["params"]["dataset"] = dataset_name
        params = vars(params)
        for p in params:
            cur_exp["params"][p] = params[p]
        cur_exp["results"] = {}

        cur_exp["test_accuracy"] = str(test_set_acc_scat)
        cur_exp["num_dims"] = str(num_dims)
        print(cur_exp)
        data[f"experiment_{n_exps}"] = cur_exp
        json.dump(data, f)

    return



if __name__ == "__main__":
    dataset_name, verbose, params = process_classification_arguments()
    print(params)

    classification_experiment(dataset_name=dataset_name, params=params, verbose=verbose)

#
