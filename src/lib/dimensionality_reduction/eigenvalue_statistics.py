###########################################################
# Methods for eigenvalue plots and statistics
# Scattering_Space/lib/dimensionality_reduction
###########################################################

import numpy as np
import pandas as pd

def compute_eigenvalue_statistics(eigenvalues, verbose=1):
    """
    Computing some statistics about the eigenvalues and variance distribution

    Args:
    -----
    eigenvalues: numpy array
       array with the eigenvalues of the covariance matrix sorted by magnitude in descending order
    verbose: integer
        verbosity level

    Returns:
    --------
    eigen_stats: dictionary
        disctionary with the computed eigenvalue and variance statistics
    """

    eigenvalues_norm = eigenvalues/np.sum(eigenvalues)
    cdf = np.cumsum(eigenvalues_norm)
    eigen_stats = {}

    # magnitude related statistics
    max_eigenvalue = np.max(eigenvalues_norm)
    mean_value = np.mean(eigenvalues_norm)
    median_value = np.median(eigenvalues_norm)
    num_eigenvalues_smaller_tenth = len(np.where(eigenvalues_norm<0.1)[0])
    num_eigenvalues_smaller_hundred = len(np.where(eigenvalues_norm<0.01)[0])
    num_eigenvalues_smaller_thousand = len(np.where(eigenvalues_norm<0.001)[0])

    if(verbose>0):
        print(f"\nMagnitude-related statistics")
        print(f"    Max. Eigenvalue: {max_eigenvalue}")
        print(f"    Mean Eigenvalue: {mean_value}")
        print(f"    Median Eigenvalue: {median_value}")
        print(f"    Number of eigenvalues < 0.1: {num_eigenvalues_smaller_tenth}")
        print(f"    Number of eigenvalues < 0.01: {num_eigenvalues_smaller_hundred}")
        print(f"    Number of eigenvalues < 0.001: {num_eigenvalues_smaller_thousand}")

    # variance related statistics
    eig_for_50_percent = int(np.where(cdf>0.5)[0][0])+1
    eig_for_80_percent = int(np.where(cdf>0.8)[0][0])+1
    eig_for_90_percent = int(np.where(cdf>0.9)[0][0])+1
    eig_for_95_percent = int(np.where(cdf>0.95)[0][0])+1
    eig_for_99_percent = int(np.where(cdf>0.99)[0][0])+1

    if(verbose>0):
        print(f"\nVariance-related statistics")
        print(f"    Number of eigenvalues for 50% variance : {eig_for_50_percent}")
        print(f"    Number of eigenvalues for 80% variance : {eig_for_80_percent}")
        print(f"    Number of eigenvalues for 90% variance : {eig_for_90_percent}")
        print(f"    Number of eigenvalues for 95% variance : {eig_for_95_percent}")
        print(f"    Number of eigenvalues for 99% variance : {eig_for_99_percent}")

    eigen_stats = {
        "max_eigenvalue": float(max_eigenvalue),
        "mean_value": float(mean_value),
        "median_value": float(median_value),
        "num_eigenvalues_smaller_tenth": num_eigenvalues_smaller_tenth,
        "num_eigenvalues_smaller_hundred": num_eigenvalues_smaller_hundred,
        "num_eigenvalues_smaller_thousand": num_eigenvalues_smaller_thousand,
        "eig_for_50_percent": eig_for_50_percent,
        "eig_for_80_percent": eig_for_80_percent,
        "eig_for_90_percent": eig_for_90_percent,
        "eig_for_95_percent": eig_for_95_percent,
        "eig_for_99_percent": eig_for_99_percent
    }

    return eigen_stats


def display_clean_stats(eigen_stats, stats="variance", index="--"):
    """
    Displaying the eigenvalue statistics in a clean manner using pandas dataframes

    Args:
    -----
    eigen_stats: dictionary or list of dictionaries
        dictionary (or dictionaries) with the statistics of a set of eigenvalues
    stats: string
        statistics to visualize ['variance', 'magnitude']
    index: list
        names to assign to each row of the pandas dataframe
    """

    assert stats in ["variance", "magnitude"]
    if type(eigen_stats) is list:
        assert len(index)==len(eigen_stats)

    # converting dictionaries into pandas dataframes
    if type(eigen_stats) is list:
        df = pd.DataFrame()
        for i, cur_stats in enumerate(eigen_stats):
            df_ = pd.DataFrame.from_dict(cur_stats, orient="index", columns=[index[i]]).transpose()
            df_.rename(index={'0': index[i]}, inplace=True)
            df = pd.concat([df, df_])
    else:
        df = pd.DataFrame.from_dict(eigen_stats, orient="index").transpose()
        df.rename(index={'0': index}, inplace=True)

    # keeping relevant columns
    if(stats == "variance"):
        col_list = ["eig_for_50_percent", "eig_for_80_percent", "eig_for_90_percent",
                    "eig_for_95_percent", "eig_for_99_percent"]
    elif(stats == "magnitude"):
        col_list = ["max_eigenvalue", "mean_value", "median_value", "num_eigenvalues_smaller_tenth",
                    "num_eigenvalues_smaller_hundred", "num_eigenvalues_smaller_thousand"]

    df = df[col_list]

    return df
