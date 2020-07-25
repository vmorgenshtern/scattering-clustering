"""
Auxiliary methods used for processing and evalution of clustering algorithms
"""

import itertools
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import sklearn.metrics as metrics


def compute_clustering_metrics(preds, labels):
    """
    Computing different clustering metrics, i.e., clustering accuracy and ARI score
    """

    acc = compute_cluster_accuracy_relaxed(predictions=preds, labels=labels)
    # acc = compute_cluster_accuracy(predictions=preds, labels=labels)
    score = metrics.adjusted_rand_score(preds, labels)
    nmi = metrics.normalized_mutual_info_score(labels_true=labels, labels_pred=preds)

    return score, acc, nmi


def compute_cluster_accuracy(predictions, labels):
    """
    Computing the clustering accuracy. THe best permutation is given by the
    Hungarian algorithm for optimum matching (This is slow)

    Args:
    -----
    predictions, labels: numpy array
        one dimensional arrays containing the prediceted and ground truth
        cluster assignments respectively
    """

    # computing permutation matrix
    D = max(predictions.max(), labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(predictions)):
        w[predictions[i], labels[i]] += 1

    # computing best permutation and accuracy figure
    ind = linear_sum_assignment(w.max() - w)
    accuracy = sum([w[i, j] for i, j in ind]) * 100 / len(predictions)

    return accuracy


def compute_cluster_accuracy_relaxed(predictions, labels):
    """
    Computing a relaxed version of the clustering accuracy. We assume that the method
    clusters with reasonable accuracy and choose as ground truth the label that is the
    most represented in each cluster (it works empirically for the tested datasets)

    Args:
    -----
    predictions, labels: numpy array
        one dimensional arrays containing the prediceted and ground truth
        cluster assignments respectively
    """

    n_correct = 0
    preds_unique = np.unique(predictions)
    for i, pred in enumerate(preds_unique):
        # obtaining predictions and targets corresponding to a given cluster
        idx = np.where(predictions==pred)[0]
        cur_preds = predictions[idx]
        lbls = labels[idx]
        # counting most represented labels
        (values,counts) = np.unique(lbls,return_counts=True)
        max_represented_lbl = values[np.argmax(counts)]
        n_correct = n_correct + np.max(counts)

    accuracy = n_correct / len(labels)

    return accuracy


def display_cluster_subset(images, pred_labels, **kwargs):
    """
    Displaying a small subset of images for each of the detected clusters
    """

    # processing optional parameters
    n_classes = kwargs["n_classes"] if("n_classes" in kwargs) else 0
    stats, labels = (kwargs["stats"], kwargs["labels"]) if "stats" in kwargs else (False, None)
    map = kwargs["label_map"] if "label_map" in kwargs else None
    if("size" in kwargs):
        cur_size = kwargs["size"]
    else:
        cur_size = (30, 5)
    if("shuffle" in kwargs and kwargs["shuffle"] is True):
        idx = np.arange(0, len(pred_labels))
        np.random.shuffle(idx)
        images, pred_labels, labels = images[idx,:], pred_labels[idx], labels[idx]

    n_labels = len(np.unique(pred_labels))
    for i in range(n_labels):
        if(n_classes > 0 and i >= n_classes):
            break
        # obtaining corresponding data
        idx = np.where(pred_labels == i)[0]
        cur_images = images[idx]
        # computing clusters stats
        if(stats is True and labels is not None):
            cur_stats = _cluster_stats(labels=labels, idx=idx, map=map)
        else:
            cur_stats = ""
        # displaying small subset
        fig, ax = plt.subplots(1,7)
        fig.set_size_inches(cur_size)
        for j in range(7):
            cur_img = cur_images[j,:]
            if("reshape" in kwargs):
                cur_img = cur_img.reshape(kwargs["reshape"])
            ax[j].imshow(cur_img)
        plt.suptitle(f"Images in cluster #{i}{cur_stats}", fontsize=18)
        plt.tight_layout()
    plt.show()

    return


def _cluster_stats(labels, idx, map=None, top_n=5):
    """
    Obtaining the frequency stats from the current cluster

    Args:
    -----
    labels: numyp array
        array with all labels from the dataset
    idx: numpy array
        indices of the samples in the current cluster
    map: dictionary
        dict that maps label index to label name
    top_n: integer
        displaying only 'top_n' classes in the stats
    """

    get_lbl_name = lambda idx: map[idx] if map is not None else idx
    labels_cluster = labels[idx]
    lbls_unique, counts = np.unique(labels_cluster, return_counts=True)

    sort_idx = np.argsort(counts)[::-1]
    counts, lbls_unique = counts[sort_idx], lbls_unique[sort_idx]
    total = len(idx)
    freq_stats = [f"#{i} {get_lbl_name(lbls_unique[i])}:{counts[i]}/{total}" for i in range(top_n)]
    stats = f"    Stats: {'     '.join(freq_stats)}"

    return stats

#
