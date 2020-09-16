"""
Auxiliary methods used for processing and evalution of clustering algorithms

@author: Angel Villar-Corrales
"""

import itertools
from tqdm import tqdm

import numpy as np
from numba import jit
from matplotlib import pyplot as plt
import sklearn.metrics as metrics


def compute_clustering_metrics(preds, labels):
    """
    Computing different clustering metrics, i.e., clustering accuracy and ARI score
    """

    acc = compute_cluster_accuracy_relaxed(predictions=preds, labels=labels)
    score = metrics.adjusted_rand_score(preds, labels)
    nmi = metrics.normalized_mutual_info_score(labels_true=labels, labels_pred=preds)

    return score, acc, nmi


def compute_cluster_accuracy_relaxed(predictions, labels):
    """
    Computing a relaxed version of the clustering accuracy. We assume that the method
    clusters with reasonable accuracy and choose as ground truth the label that is the
    most represented in each cluster.

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


def compute_cluster_accuracy(predictions, labels):
    """
    Computed the accuracy on the label of the predicted clusters by computing all permutations
    and keeping the one with the highest accuracy

    Args:
    -----
    predictions, labels: numpy array
        one dimensional arrays containing the prediceted and ground truth
        cluster assignments respectively

    Returns:
    --------
    accuracy: float
        percentage of correct labels
    permutation: tuple
        tuple with the correct label permutation
    """

    # shifting labels so that they are in the range [0,N-1]
    unique_lbl = np.unique(labels)
    n_labels = len(unique_lbl)
    for i in range(n_labels):
        idx = np.where(labels==unique_lbl[i])[0]
        labels[idx] = i
    # unique_lbl = np.unique(labels)
    unique_lbl = list(set(labels))

    # computing the accuracy for all permutations
    accuracies = []
    permutations = list(itertools.permutations(unique_lbl))

    for i, permutation in enumerate(tqdm(permutations)):
        cur_lbls = permute_labels(labels, permutation)
        correct_labels = len(np.where(predictions == cur_lbls)[0])
        cur_acc = 100 * correct_labels / len(predictions)
        accuracies.append(cur_acc)

    accuracies = np.array(accuracies)
    accuracy = np.max(accuracies)
    permutation_idx = np.argmax(accuracies)
    permutation = permutations[permutation_idx]

    return accuracy, permutation


def compute_cluster_accuracy_given_permutations(predictions, labels, permutation):
    """
    Permuting the labels using the argument tuple and computing the label accuracy

    Args:
    -----
    predictions, labels: numpy array
        one dimensional arrays containing the prediceted and ground truth
        cluster assignments respectively
    permutation: tuple
        tuple with the current label permutation

    Returns:
    --------
    accuracy: float
        percentage of correct labels
    """

    # permuting labels using input tuple
    cur_lbls = permute_labels(labels, permutation)

    # computing label accuracy
    correct_labels = len(np.where(predictions == cur_lbls)[0])
    accuracy = 100 * correct_labels / len(predictions)

    return accuracy


# @jit(nopython=True)
def permute_labels(labels, permutation):
    """
    Permuting the labels given the permutation factor

    Args:
    -----
    labels: numpy array
        one dimensional numpy array containing the labels to permute
    permutation: tuple
        tuple with the current label permutation

    Example:
    --------
        labels = [0,0,1,1,2,2]; perm=(1,2,0)
        permuted_labels = [1,1,2,2,0,0]
    """

    n_labels = len(np.unique(labels))
    permuted_labels = np.copy(labels)

    for i, p in enumerate(permutation):
        idx = np.where(labels==i)[0]
        permuted_labels[idx] = p

    return permuted_labels


def display_cluster_subset(images, pred_labels, **kwargs):
    """
    Displaying a small subset of images for each of the detected clusters
    """

    if("size" in kwargs):
        cur_size = kwargs["size"]
    else:
        cur_size = (30, 5)

    n_labels = len(np.unique(pred_labels))

    for i in range(n_labels):
        fig, ax = plt.subplots(1,7)
        fig.set_size_inches(cur_size)

        idx = np.where(pred_labels == i)[0]
        cur_images = images[idx]
        for j in range(7):
            cur_img = cur_images[j,:].reshape(28,28)
            ax[j].imshow(cur_img)
        plt.suptitle(f"Images in cluster #{i}", fontsize=18)
        plt.tight_layout()
    plt.show()

    return

#
