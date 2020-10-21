"""
Performing image clustering by applyng spectral clustering (with POC preprocessing)
to the scattering representations of the original small images

@author: Angel Villar-Corrales
"""

import os
import json
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from lib.clustering.uspec import USPEC
from lib.clustering.utils import compute_clustering_metrics
from lib.data.data_loading import ClassificationDataset
from lib.data.data_processing import convert_images_to_scat
from lib.dimensionality_reduction import pca
from lib.projections.POC import POC
from lib.scattering.scattering_methods import scattering_layer
from lib.utils.arguments import process_clustering_arguments
from lib.utils.logger import print_, Logger, log_function
from lib.utils.utils import create_directory
from CONFIG import CONFIG


@log_function
def clustering_experiment(dataset_name, params, verbose=0, random_seed=0):
    """
    Performing a clustering experiment (with possible POC preprocessing) on scattering
    features of the original images
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading data and breaking it into different dataset splits
    dataset = ClassificationDataset(data_path=CONFIG["paths"]["data_path"],
                                    dataset_name=dataset_name,
                                    valid_size=0)
    imgs, labels = dataset.get_all_data()

    # computing scattering representations
    t0 = time.time()
    if(params.scattering):
        scattering_net, _ = scattering_layer(J=params.J, shape=(params.shape, params.shape),
                                             max_order=params.max_order, L=params.L)
        scattering_net = scattering_net.cuda() if device.type == 'cuda' else scattering_net
        print_("Computing scattering features for dataset...", verbose=verbose)
        data  = convert_images_to_scat(imgs, scattering=scattering_net,
                                       device=device, equalize=params.equalize,
                                       batch_size=params.batch_size,
                                       pad_size=params.shape)
    else:
        data = imgs
    n_labels = len(np.unique(labels))

    # reducing dimensionality of scattering features using pca
    if(params.pca == True):
        print_(f"Reducidng dimensionality using PCA to {params.pca_dims}", verbose=verbose)
        n_feats = data.shape[0]
        data = data.reshape(n_feats, -1)
        data = pca(data=data, target_dimensions=params.pca_dims)

    # POC preprocessing: removing the top directions of variance as a preprocessing step
    t1 = time.time()
    if(params.poc_preprocessing):
        print_(f"POC algorithm removing {params.num_dims} directions", verbose=verbose)
        poc = POC()
        poc.fit(data=data)
        proj_data = poc.transform(data=data, n_dims=params.num_dims)
    else:
        proj_data = data.reshape(data.shape[0], -1)

    # clustering using Ultra-Scalable Spectral Clustering
    t2 = time.time()
    if(params.uspec):
        print_(f"Clustering using U-SPEC", verbose=verbose)
        uspec = USPEC(p_interm=params.num_candidates, p_final=params.num_reps,
                      n_neighbors=5, num_clusters=params.num_clusters,
                      num_iters=100, random_seed=random_seed)
        preds = uspec.cluster(data=proj_data, verbose=verbose)
    else:
        print_(f"Clustering using K-Means", verbose=verbose)
        kmeans = KMeans(n_clusters=params.num_clusters, random_state=random_seed)
        kmeans = kmeans.fit(proj_data)
        preds = kmeans.labels_
    t3 = time.time()

    cluster_score, cluster_acc, cluster_nmi = compute_clustering_metrics(preds=preds,
                                                                         labels=labels)
    print_(f"Clustering Accuracy: {round(cluster_acc,3)}", verbose=verbose)
    print_(f"Clustering ARI Score: {round(cluster_score,3)}", verbose=verbose)
    print_(f"Clustering ARI Score: {round(cluster_nmi,3)}", verbose=verbose)

    # loading previous results, if any
    results_path = os.path.join(os.getcwd(), CONFIG["paths"]["results_path"])
    _ = create_directory(results_path)
    if(params.fname != None and len(params.fname) > 0 and params.fname[-5:]==".json"):
        fname = params.fname
    else:
        poc = "poc" if params.poc_preprocessing==True else "not-poc"
        fname =  f"{dataset_name}_{poc}_clustering_results.json"
    results_file = os.path.join(results_path, fname)
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
        cur_exp["params"]["random_seed"] = random_seed
        params = vars(params)
        for p in params:
            cur_exp["params"][p] = params[p]
        cur_exp["results"] = {}
        cur_exp["results"]["cluster_acc"] = round(cluster_acc,3)
        cur_exp["results"]["cluster_ari"] = round(cluster_score,3)
        cur_exp["results"]["cluster_nmi"] = round(cluster_nmi,3)
        cur_exp["timing"] = {}
        cur_exp["timing"]["scattering"] = t1 - t0
        cur_exp["timing"]["preprocessing"] = t2 - t1
        cur_exp["timing"]["clustering"] = t3 - t2
        cur_exp["timing"]["total"] = t3 - t0
        print_(cur_exp, verbose=verbose)
        data[f"experiment_{n_exps}"] = cur_exp
        json.dump(data, f)

    return


if __name__ == "__main__":
    os.system("clear")
    logger = Logger(exp_path=os.getcwd())

    dataset_name, verbose, random_seed, params = process_clustering_arguments()
    print_(params, message_type="new_exp")

    clustering_experiment(dataset_name=dataset_name, params=params,
                          verbose=verbose, random_seed=random_seed)

#
