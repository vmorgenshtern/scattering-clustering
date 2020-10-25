"""
Methods for extractig and filtering the results from the 'clustering_results.json'

@author: Angel Villar-Corrales
"""

import os
import json
import argparse

import numpy as np

from CONFIG import CONFIG


def process_arguments():
    """
    Processing command line arguments
    """

    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--fname', help="Name of the file with current results to process")
    parser.add_argument('--outfile', help="Name of the file where the results will be stored")
    parser.add_argument('--dataset_name', help="Name of the dataset to classify or " \
                        "cluster: ['mnist', 'fashion_mnist', 'svhn', 'usps', 'mnist-test']",
                        default="mnist")
    parser.add_argument('--scattering', help="If True, clustering is performend on "\
                        "scattering feature reperesentation. Otherwise on images")
    parser.add_argument('--poc_preprocessing', help="if True, the POC algorithm is a "\
                        "preprocessing step to the clustering algorithm")
    parser.add_argument('--uspec', help="If True, U-SPEC algorithm is used for clustering. " \
                        "Otherwise K-Means performs the clustering")

    args = parser.parse_args()

    fname = args.fname
    outfile = args.outfile
    dataset_name = args.dataset_name
    scattering = (args.scattering == "True")
    poc_preprocessing = (args.poc_preprocessing == "True")
    uspec = (args.uspec == "True")

    # checking that file with input results exists
    fpath = os.path.join(CONFIG["paths"]["results_path"], fname)
    assert os.path.exists(fpath), f"Given file: {fname} does not exists in 'results' directory"
    # checking that output file (if given) exists and is a json file
    if(outfile is not None):
        outpath = os.path.join(CONFIG["paths"]["results_path"], outfile)
        assert outfile[-5:] == ".json"
    # checking correctness in other parameters
    assert dataset_name in ["mnist", "fashion_mnist", 'usps', 'mnist-test'],\
        f"ERROR! wrong 'dataset_name' parameter: {dataset_name}.\n Only ['mnist',"\
        f"'fashion_mnist', 'usps', 'mnist-test'] are allowed"

    return fname, outfile, dataset_name, scattering, poc_preprocessing, uspec



def extract_results(fname, outfile, dataset_name, scattering, poc_preprocessing, uspec):
    """
    Loading the results corresponding to the parameters. Extracting results and
    running times for all runs with the parameters. Computing mean and std_dev values
    """

    # loading data
    fpath = os.path.join(CONFIG["paths"]["results_path"], fname)
    if(not os.path.exists(fpath)):
        print(f"ERROR! Results file: {fname} does not exists...")
        exit()
    with open(fpath) as f:
        experiments = json.load(f)

    # extracting relevant inforamtion
    acc, nmi, ari = [], [], []
    scat_time, prep_time, cluster_time, total_time = [], [], [], []
    for exp_key in experiments:
        cur_exp = experiments[exp_key]
        params = cur_exp["params"]
        # checking that fields 'scat..', 'poc_prep..' & 'uspec' values match parameters
        if(params["dataset"] != dataset_name or params["scattering"] != scattering or
           params["poc_preprocessing"] != poc_preprocessing or params["uspec"] != uspec):
           continue
        acc.append(cur_exp["results"]["cluster_acc"])
        ari.append(cur_exp["results"]["cluster_ari"])
        nmi.append(cur_exp["results"]["cluster_nmi"])
        scat_time.append(cur_exp["timing"]["scattering"])
        prep_time.append(cur_exp["timing"]["preprocessing"])
        cluster_time.append(cur_exp["timing"]["clustering"])
        total_time.append(cur_exp["timing"]["total"])
    if(len(acc) == 0):
        print(f"No expriments found in file: {fname} matching params:\n" \
              f"    dataset_name: {dataset_name}\n    scattering: {scattering}\n" \
              f"    poc_preprocessing: {poc_preprocessing}\n    uspec:{uspec}")
        exit()

    # measuring mean and variance for all metrics
    compute_mean_var = lambda x: (np.mean(x), np.std(x))
    acc_mean, acc_std = compute_mean_var(acc)
    nmi_mean, nmi_std = compute_mean_var(nmi)
    ari_mean, ari_std = compute_mean_var(ari)
    scat_time_mean, scat_time_std = compute_mean_var(scat_time)
    prep_time_mean, prep_time_std = compute_mean_var(prep_time)
    cluster_time_mean, cluster_time_std = compute_mean_var(cluster_time)
    time_mean, time_std = compute_mean_var(total_time)

    # saving final results
    if(outfile is None):
        outfile = "_paper_results.json"
    results_file = os.path.join(CONFIG["paths"]["results_path"], outfile)
    if(os.path.exists(results_file)):
        final_data = json.load(open(results_file))
    else:
        final_data = {}

    exp_name = f"dataset_{dataset_name}_scattering_{scattering}_" \
               f"poc_{poc_preprocessing}_uspec_{uspec}"
    params = {
        "dataset_name": dataset_name,
        "scattering": scattering,
        "poc_preprocessing": poc_preprocessing,
        "uspec": uspec,
        "fname": fname
    }
    results = {
        "acc_mean": acc_mean, "acc_std": acc_std,
        "ari_mean": ari_mean, "ari_std": ari_std,
        "nmi_mean": nmi_mean, "nmi_std": nmi_std
     }
    timing = {
        "scattering_mean": scat_time_mean, "scattering_std": scat_time_std,
        "preprocessing_mean": prep_time_mean, "preprocessing_std": prep_time_std,
        "clustering_mean": cluster_time_mean, "clustering_std": cluster_time_std,
        "total_mean": time_mean, "total_std": time_std,
    }
    final_data[exp_name] = {
        "params": params,
        "results": results,
        "timing": timing
    }

    with open(results_file, "w") as f:
        json.dump(final_data, f)

    return


if __name__ == "__main__":
    os.system("clear")
    fname, outfile, dataset_name, scattering, poc_preprocessing, uspec = process_arguments()
    extract_results(fname=fname, outfile=outfile, dataset_name=dataset_name,
                    scattering=scattering, poc_preprocessing=poc_preprocessing, uspec=uspec)

#
