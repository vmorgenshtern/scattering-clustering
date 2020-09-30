"""
Methods for extractig and filtering the results from the 'clustering_results.json'

@author: Angel Villar-Corrales
"""

import os
import json
import argparse

import numpy as np


def process_arguments():
    """
    Processing command line arguments
    """

    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--dataset_name', help="Name of the dataset to classify or " \
                        "cluster: ['mnist', 'fashion_mnist', 'svhn', 'usps', 'mnist-test']",
                        default="mnist")
    parser.add_argument('--poc_preprocessing', help="if True, the POC algorithm is a "\
                        "preprocessing step to the clustering algorithm")
    parser.add_argument('--method', help="baseline clustering method: ['k_means', " \
                        "'spectral_clustering', 'dbscan', 'pipeline']", default="k_means")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    method = args.method
    poc_preprocessing = args.poc_preprocessing

    poc_preprocessing = (poc_preprocessing == "True") if poc_preprocessing != None else None
    assert dataset_name in ["mnist", "fashion_mnist", 'svhn', 'usps', 'mnist-test', \
                            'coil-100', 'cifar'], \
        f"ERROR! wrong 'dataset_name' parameter: {dataset_name}.\n Only ['mnist',"\
        f"'fashion_mnist', 'usps', 'mnist-test', 'cifar'] are allowed"
    assert method in ["k_means", "spectral_clustering", "dbscan", "pipeline"]


    return dataset_name, method, poc_preprocessing


def extract_results(dataset_name, method, poc_preprocessing):
    """
    Loading the results corresponding to the parameters. Extracting results and
    running times for all runs with the parameters. Computing mean and std_dev values
    """

    # loading data
    if(method == "pipeline"):
        poc_flag = "poc" if poc_preprocessing == True else ""
        fname = f"{dataset_name}_{poc_flag}_clustering_results.json"
    else:
        fname = f"{method}_{dataset_name}_results.json"
    fpath = os.path.join(os.getcwd(), "results", fname)
    if(not os.path.exists(fpath)):
        print(f"ERROR! Results file: {fname} does not exists...")
        exit()
    with open(fpath) as f:
        experiments = json.load(f)

    # extracting relevant inforamtion
    acc, nmi, ari, time = [], [], [], []
    for exp_key in experiments:
        cur_exp = experiments[exp_key]
        acc.append(cur_exp["results"]["cluster_acc"])
        ari.append(cur_exp["results"]["cluster_ari"])
        nmi.append(cur_exp["results"]["cluster_nmi"])
        time.append(cur_exp["timing"]["total"])

    acc_mean, acc_std = np.mean(acc), np.std(acc)
    nmi_mean, nmi_std = np.mean(nmi), np.std(nmi)
    ari_mean, ari_std = np.mean(ari), np.std(ari)
    time_mean, time_std = np.mean(time), np.std(time)

    # saving final results
    fname = os.path.join(os.getcwd(), "results", "paper_results.json")
    if(os.path.exists(fname)):
        final_data = json.load(open(fname))
    else:
        final_data = {}

    if(method == "pipeline" and poc_preprocessing == True):
        method = "pipeline-poc"
    elif(method == "pipeline" and poc_preprocessing != True):
        method = "pipeline"
    final_data[f"{dataset_name}_{method}"] = {}
    final_data[f"{dataset_name}_{method}"]["acc_mean"] = acc_mean
    final_data[f"{dataset_name}_{method}"]["acc_std"] = acc_std
    final_data[f"{dataset_name}_{method}"]["ari_mean"] = ari_mean
    final_data[f"{dataset_name}_{method}"]["ari_std"] = ari_std
    final_data[f"{dataset_name}_{method}"]["nmi_mean"] = nmi_mean
    final_data[f"{dataset_name}_{method}"]["nmi_std"] = nmi_std
    final_data[f"{dataset_name}_{method}"]["time_mean"] = time_mean
    final_data[f"{dataset_name}_{method}"]["time_std"] = time_std
    with open(fname, "w") as f:
        json.dump(final_data, f)

    return


if __name__ == "__main__":
    os.system("clear")
    dataset_name, method, poc_preprocessing = process_arguments()
    extract_results(dataset_name, method, poc_preprocessing)

#
