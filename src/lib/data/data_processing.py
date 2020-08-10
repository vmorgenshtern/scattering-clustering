"""
Utils for data preprocessing and handling

@author: Angel Villar-Corrales
"""

from tqdm import tqdm

import numpy as np
import torch
from lib.scattering.equalizations import max_norm_equalization
from lib.data.custom_transforms import pad_img


def get_classwise_data(data, labels, label=0, verbose=0):
    """
    Obtaining features and labels corresponding to a particular class

    Args:
    -----
    data: numpy array
        array-like object with the data
    labels: np array or list
        arraylike object with the labels corresponding to the data
    label: integer
        label corresponding to the data that we want to extract
    verbose: integer
        verbosity level

    Returns:
    --------
    classwise_data: np array
        array with the data corresponding to the desired class
    """

    idx = np.where(labels==label)[0]
    classwise_data = data[idx,:]
    if(verbose>0):
        print(f"There are {len(idx)} datapoints with label {label}")

    return classwise_data


def remove_class_data(data, labels, label=0, verbose=1):
    """
    Obtaining features and labels except the ones of a particular class

    Args:
    -----
    data: numpy array
        array-like object with the data
    labels: np array or list
        arraylike object with the labels corresponding to the data
    label: integer
        label corresponding to the data that we want to remove
    verbose: integer
        verbosity level

    Returns:
    --------
    classwise_data: np array
        array with the data having removed the desired class
    """

    idx = np.where(labels!=label)[0]
    classwise_data = data[idx,:]
    classwise_labels = labels[idx]
    if(verbose>0):
        print(f"There are {len(idx)} datapoints with label different than {label}")

    return classwise_data, classwise_labels



def convert_images_to_scat(images, scattering, device, equalize=False):
    """
    Args:
    -----
    images: numpy array/torch Tensor
        Array (B,C,H,W) containing the images to convert to the scattering domain
    scattering: kymatio network
        kymation scattering network used top process the images
    device: device
        cpu or gpu
    equalize: Boolean
        if True, scattering features are max-equalized

    Returns:
    --------
    scattering: numpy arrays
        scattering features from the images in the data loader
    """

    # preprocessing
    if(len(images.shape) == 3):
        images = images[:, np.newaxis, :, :]
    if(isinstance(images, np.ndarray)):
        images = torch.Tensor(images).to(device)
    padded_imgs = pad_img(images, target_shape=(32,32)).squeeze()
    padded_imgs = padded_imgs.to(device)

    # scattering forward
    step = 64
    scat_features = []
    for i in tqdm(range(0, padded_imgs.shape[0], step)):
        cur_scat_features = scattering(padded_imgs[step*i:step*(i+1),:])
        scat_features.append(cur_scat_features)
    scat_features = np.concatenate(scat_features, axis=0)

    scat_features = np.array(scat_features.cpu(), dtype=np.float64)

    if(equalize):
        scat_features, _ = max_norm_equalization(scat_features)

    return scat_features


#
