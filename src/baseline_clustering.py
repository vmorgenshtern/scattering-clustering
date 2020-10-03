"""
Image clustering using some of the traditional methods for baseline purposes

@author: Angel Villar-Corrales
"""

import os
import json
import time

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

from lib.clustering.utils import compute_clustering_metrics
from lib.data.data_loading import ClassificationDataset
from lib.utils.arguments import process_baseline_arguments
from lib.utils.utils import create_directory
from CONFIG import CONFIG


def baseline_clustering(dataset_name, method, verbose, random_seed):
    """
    Performing image clustering using one of the baseline methods

    Args:
    -----
    dataset_name: string
        name of the dataset to perform clustering to
    method: string
        name of the method to use for clustering
    verbose: boolean
        verbosity level
    """

    # loading data and breaking it into different dataset splits
    dataset = ClassificationDataset(data_path=CONFIG["paths"]["data_path"],
                                    dataset_name=dataset_name,
                                    valid_size=0)
    imgs, labels = dataset.get_all_data()
    n_imgs = len(labels)
    imgs = imgs.reshape(n_imgs, -1)

    # clustering dataset samples
    t0 = time.time()
    if(method == "k_means"):
        clusterer = KMeans(n_clusters=10, random_state=random_seed,
                           verbose=verbose)
        clusterer = clusterer.fit(imgs)

    elif(method == "spectral_clustering"):
        clusterer = SpectralClustering(n_clusters=10, random_state=random_seed,
                                       affinity='nearest_neighbors', n_neighbors=5)
        clusterer = clusterer.fit(imgs)

    elif(method == "dbscan"):
        clusterer = DBSCAN(eps=3, min_samples=2)
        clusterer = clusterer.fit(imgs)

    preds = clusterer.labels_
    t1 = time.time()


    cluster_score, cluster_acc, cluster_nmi = compute_clustering_metrics(preds=preds,
                                                                         labels=labels)
    print(f"Clustering Accuracy: {round(cluster_acc,3)}")
    print(f"Clustering ARI Score: {round(cluster_score,3)}")
    print(f"Clustering NMI Score: {round(cluster_nmi,3)}")


    # loading previous results, if any
    results_path = os.path.join(os.getcwd(), CONFIG["paths"]["results_path"])
    _ = create_directory(results_path)
    results_file = os.path.join(results_path, f"{method}_{dataset_name}_results.json")
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
        cur_exp["params"]["method"] = method
        cur_exp["params"]["random_seed"] = random_seed
        cur_exp["results"] = {}
        cur_exp["results"]["cluster_acc"] = round(cluster_acc,3)
        cur_exp["results"]["cluster_ari"] = round(cluster_score,3)
        cur_exp["results"]["cluster_nmi"] = round(cluster_nmi,3)
        cur_exp["timing"] = {}
        cur_exp["timing"]["total"] = t1 - t0
        data[f"experiment_{n_exps}"] = cur_exp
        json.dump(data, f)

    return


if __name__ == "__main__":
    os.system("clear")
    dataset_name, method, verbose, random_seed = process_baseline_arguments()
    baseline_clustering(dataset_name=dataset_name,
                        method=method,
                        verbose=verbose,
                        random_seed=random_seed)


#
