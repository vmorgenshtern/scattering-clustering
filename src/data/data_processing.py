"""
Utils for data preprocessing and handling
"""

import numpy as np
import torch
from lib.scattering.equalizations import max_norm_equalization


def get_classwise_data(data, labels, label=0, verbose=1):
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



def compute_scat_features(data_loader, scattering, device):
    """
    Converting images from a data loader to scattering features

    Args:
    -----
    data_loader: torchvision data_loader
        Data loader fit with the images to process
    scattering: kymatio network
        kymation scattering network used top process the images
    device: device
        cpu or gpu

    Returns:
    --------
    images: numpy array
        array containing all images from the data loader
    scattering: numpy arrays
        scattering features from the images in the data loader
    labels: numpy array
        array listing the labels for each image
    """

    images = []
    scat_features = []
    labels = []

    for i,(img, label) in enumerate(tqdm(data_loader)):
        # img = custom_transforms.pad_mnist(img)
        img = custom_transforms.pad_img(img, target_shape=(32,32)).squeeze()
        img = img.to(device)
        img = img.squeeze()

        cur_coeffs = scattering(img)

        cur_coeffs = np.array(cur_coeffs.cpu(), dtype=np.float64)
        scat_features.append(cur_coeffs)
        images.append(np.array(img.cpu(), dtype=np.float64))
        labels.append(np.array(label))

    scat_features = np.concatenate(scat_features, axis=0)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    return images, scat_features, labels


def convert_images_to_scat(images, scattering, device, equalize=False):
    """
    Converting an array of images to the scattering domain

    Args:
    -----
    images: numpy array
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
    images = torch.Tensor(images).to(device)
    padded_imgs = custom_transforms.pad_img(images, target_shape=(32,32)).squeeze()
    padded_imgs = padded_imgs.to(device)

    # scattering forward
    scat_features = scattering(padded_imgs)
    scat_features = np.array(scat_features.cpu(), dtype=np.float64)

    if(equalize):
        scat_features, _ = max_norm_equalization(scat_features)

    return scat_features


#
