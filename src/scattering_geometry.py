"""
Computing plots to display the geometry and distribution of the scatterin transform

@author: Angel Villar-Corrales
"""

import os

import numpy as np
import torch
from  matplotlib import pyplot as plt

from lib.data.data_loading import ClassificationDataset
from lib.data.data_processing import convert_images_to_scat
from lib.projections.dimensionality_reduction import compute_eigendecomposition
from lib.projections.principal_angles import compute_principal_angles, \
    compute_angle_statistics, display_principal_angle_statistics, subspace_affinity
from lib.projections.projection_orthogonal_complement import get_classwise_data
from lib.scattering.scattering_methods import scattering_layer
from lib.utils.visualizations import compute_eigenvalue_histogram, \
    display_principal_angles_histogram
from CONFIG import CONFIG


def compute_scattering_geometrical_plots():
    """
    Computing different plots illustrating the fact that the scattering transform
    linearizes the variance accross a handfull of directions. Furthermore, these
    few directions are highly correlated.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading the MNIST dataset
    dataset_name = "mnist-test"
    dataset = ClassificationDataset(data_path=CONFIG["paths"]["data_path"],
                                    dataset_name=dataset_name,
                                    valid_size=0)
    imgs, labels = dataset.get_all_data()

    # computing scattering representations of complete MNIST dataset
    scattering_net, _ = scattering_layer(J=3, shape=(32, 32),
                                         max_order=2, L=6)
    scattering_net = scattering_net.cuda() if device.type == 'cuda' else scattering_net
    scat_features  = convert_images_to_scat(imgs, scattering=scattering_net,
                                            device=device, equalize=True,
                                            batch_size=512,
                                            pad_size=32)

    # selecting two classes (0 and 2) for visualization purposes
    imgs_0 = get_classwise_data(data=imgs, labels=labels, label=0, verbose=True)
    imgs_2 = get_classwise_data(data=imgs, labels=labels, label=2, verbose=True)

    # converting classwise images to scattering domain
    scat_features_0  = convert_images_to_scat(imgs_0, scattering=scattering_net,
                                              device=device, equalize=True,
                                              batch_size=256, pad_size=32)
    scat_features_2  = convert_images_to_scat(imgs_2, scattering=scattering_net,
                                              device=device, equalize=True,
                                              batch_size=256, pad_size=32)

    # computing eigendecomposition of imgs/scat_features
    e_vals_img, e_vect_img = compute_eigendecomposition(data=imgs, standarize=True)
    e_vals_scat, e_vect_scat = compute_eigendecomposition(data=scat_features, standarize=True)

    e_vals_img_0, e_vect_img_0 = compute_eigendecomposition(data=imgs_0, standarize=True)
    e_vals_img_2, e_vect_img_2 = compute_eigendecomposition(data=imgs_2, standarize=True)
    e_vals_scat_0, e_vect_scat_0 = compute_eigendecomposition(data=scat_features_0, standarize=True)
    e_vals_scat_2, e_vect_scat_2 = compute_eigendecomposition(data=scat_features_2, standarize=True)

    # computing eigenvalue histograms for the different data subsets
    savepaths = ["eigenvalues_mnist_img.png", "eigenvalues_mnist_scat.png", "mnist_test_img_0.png",
                 "mnist_test_scat_0.png", "mnist_test_img_2.png", "mnist_test_scat_2.png"]
    eigenvalues = [e_vals_img, e_vals_scat, e_vals_img_0,
                   e_vals_scat_0, e_vals_img_2, e_vals_scat_2]
    for i in range(len(savepaths)):
        savepath = os.path.join(CONFIG["paths"]["plots_path"], savepaths[i])
        _ = compute_eigenvalue_histogram(eigenvalues=eigenvalues[i], savefig=savepath,
                                         title="")

    # computing the principal angles between class 0 and class 2
    # both in the image and in the scattering domain
    n_angles = 10
    principal_angles_img = compute_principal_angles(P=e_vect_img_0,
                                                    Q=e_vect_img_2,
                                                    n_dims=n_angles)
    principal_angles_scat = compute_principal_angles(P=e_vect_scat_0,
                                                     Q=e_vect_scat_2,
                                                     n_dims=n_angles)
    stats_img = compute_angle_statistics(principal_angles_img)
    stats_scat = compute_angle_statistics(principal_angles_scat)
    stats_df_img = display_principal_angle_statistics(stats_img)
    stats_df_scat = display_principal_angle_statistics(stats_scat)
    affinities_img = [subspace_affinity(principal_angles_img[:i]) for i in range(1,10)]
    affinities_scat = [subspace_affinity(principal_angles_scat[:i]) for i in range(1,10)]

    #
    savepath = os.path.join(CONFIG["paths"]["plots_path"], "principal_angles_img.png")
    _ = display_principal_angles_histogram(principal_angles_img,
                                           savefig=savepath, corrs=False,
                                           yscale="linear",
                                           xlabel="Angle (radians)",
                                           ylabel="Number of angles")
    savepath = os.path.join(CONFIG["paths"]["plots_path"], "principal_angles_scat.png")
    _ = display_principal_angles_histogram(principal_angles_scat,
                                           savefig=savepath, corrs=False,
                                           yscale="linear",
                                           xlabel="Angle (radians)",
                                           ylabel="Number of angles")

    print("\nImage Principal Angles")
    print(principal_angles_img)
    print(stats_df_img)
    print("\n\nScattering Principal Angles")
    print(principal_angles_scat)
    print(stats_df_scat)
    print("\nSubspace Affinities Img")
    print(affinities_img)
    print("\nSubspace Affinities Scat")
    print(affinities_scat)

    return


if __name__ == "__main__":
    os.system("clear")
    compute_scattering_geometrical_plots()

#
