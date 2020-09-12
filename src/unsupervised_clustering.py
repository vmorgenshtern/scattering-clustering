"""
Performing image clustering by applyng spectral clustering (with POC preprocessing)
to the scattering representations of the original small images

@author: Angel Villar-Corrales
"""

import os
import json

import numpy as np
import torch

from lib.clustering.uspec import USPEC
from lib.clustering.utils import compute_clustering_metrics
from lib.data.data_loading import ClassificationDataset
from lib.data.data_processing import convert_loader_to_scat
from lib.projections.POC import POC
from lib.scattering.scattering_methods import scattering_layer
from lib.utils.arguments import process_classification_arguments
from lib.utils.utils import create_directory
from CONFIG import CONFIG


def clustering_experiment(dataset_name, params, verbose=0):
    """
    Performing a clustering experiment (with possible POC preprocessing) on scattering
    features of the original images
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading data and breaking it into different dataset splits
    dataset = ClassificationDataset(data_path=CONFIG["paths"]["data_path"],
                                    dataset_name=dataset_name,
                                    valid_size=0)
    train_loader = dataset.get_data_loader(split="train", batch_size=params.batch_size)
    test_loader = dataset.get_data_loader(split="test", batch_size=params.batch_size)

    # computing scattering representations
    scattering_net, _ = scattering_layer()
    scattering_net = scattering_net.cuda() if device.type == 'cuda' else scattering_net
    if verbose > 0: print("Computing scattering features for training set...")
    train_imgs, train_scat_features, train_labels = \
        convert_loader_to_scat(train_loader, scattering=scattering_net,
                               device=device, equalize=params.equalize,
                               verbose=verbose)
    if verbose > 0: print("Computing scattering features for test set...")
    test_imgs, test_scat_features, test_labels = \
        convert_loader_to_scat(test_loader, scattering=scattering_net,
                               device=device, equalize=params.equalize,
                               verbose=verbose)
    n_labels = len(np.unique(train_labels))

    # POC preprocessing: removing the top directions of variance as a preprocessing step
    poc = POC()
    poc.fit(data=train_scat_features)
    proj_data = poc.transform(data=train_scat_features, n_dims=params.num_dims)

    # clustering using Ultra-Scalable Spectral Clustering
    uspec = USPEC(p_interm=1e4, p_final=1e3, n_neighbors=5, num_clusters=10, num_iters=100)
    cur_preds = uspec.cluster(data=proj_data, verbose=verbose)

    cluster_score, cluster_acc = compute_clustering_metrics(preds=preds, labels=all_labels)
    print(f"Clustering Accuracy: {round(cluster_acc*100,2)}")
    print(f"Clustering ARI Score: {round(cluster_score,2)}")

    # loading previous results, if any
    results_path = os.path.join(os.getcwd(), CONFIG["paths"]["results_path"])
    _ = create_directory(results_path)
    results_file = os.path.join(results_path, "clustering_results.json")
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

        cur_exp["cluster_score"] = str(cluster_score)
        cur_exp["cluster_acc"] = str(cluster_acc)
        print(cur_exp)
        data[f"experiment_{n_exps}"] = cur_exp
        json.dump(data, f)

    return


if __name__ == "__main__":
    dataset_name, verbose, params = process_clustering_arguments()
    print(params)

    clustering_experiment(dataset_name=dataset_name, params=params, verbose=verbose)

#
