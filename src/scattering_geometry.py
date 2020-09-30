"""
Computing plots to display the geometry and distribution of the scatterin transform

@author: Angel Villar-Corrales
"""

import os

import numpy as np
from  matplotlib import pyplot as plt

from lib.data.data_loading import ClassificationDataset
from lib.data.data_processing import convert_images_to_scat
from lib.projection_orthogonal_complement import get_classwise_data
from lib.projections.dimensionality_reduction import compute_eigendecomposition
from lib.scattering.scattering_methods import scattering_layer


def compute_scattering_geometrical_plots():
    """
    Computing different plots illustrating the fact that the scattering transform
    linearizes the variance accross a handfull of directions. Furthermore, these
    few directions are highly correlated.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading the MNIST dataset
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
                                            batch_size=64,
                                            pad_size=32)

    # selecting two classes (0 and 4) for visualization purposes
    imgs_0 = get_classwise_data(data=imgs, labels=labels, label=0, verbose=True)
    imgs_4 = get_classwise_data(data=imgs, labels=labels, label=4, verbose=True)

    # converting classwise images to scattering domain
    scat_features_0  = convert_images_to_scat(data_0, scattering=scattering_net,
                                              device=device, equalize=True,
                                              batch_size=64, pad_size=32)
    scat_features_4  = convert_images_to_scat(data_4, scattering=scattering_net,
                                              device=device, equalize=True,
                                              batch_size=64, pad_size=32)

    # computing eigendecomposition of imgs/scat_features
    e_vals_img, e_vect_img = compute_eigendecomposition(data=imgs, standarize=True)
    e_vals_scat, e_vect_scat = compute_eigendecomposition(data=scat_features, standarize=True)

    e_vals_img_0, e_vect_img_0 = compute_eigendecomposition(data=imgs_0, standarize=True)
    e_vals_img_4, e_vect_img_4 = compute_eigendecomposition(data=imgs_4, standarize=True)
    e_vals_scat_0, e_vect_scat_0 = compute_eigendecomposition(data=scat_features_0, standarize=True)
    e_vals_scat_4, e_vect_scat_4 = compute_eigendecomposition(data=scat_features_4, standarize=True)


    return


if __name__ == "__main__":
    os.system("clear")
    compute_scattering_geometrical_plots()

#
