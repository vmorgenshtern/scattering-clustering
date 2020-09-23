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



def convert_images_to_scat(images, scattering, device, equalize=False,
                           batch_size=None, pad_size=32):
    """
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
    batch_size: integer
        if positive and not None, scattering transform is computed in a batch-wise manner

    Returns:
    --------
    scattering: numpy arrays
        scattering features from the images in the data loader
    """

    # preprocessing
    images = torch.Tensor(images).to(device)
    if(pad_size > 0):
        padded_imgs = pad_img(images, target_shape=(pad_size,pad_size)).squeeze()
    else:
        padded_imgs = images.squeeze()
    padded_imgs = padded_imgs.to(device)

    # scattering forward
    if(batch_size is not None and batch_size > 0):
        # computing scattering in a batchwise fashion => use this for large datasets
        scat_features = []
        n_batches = int(np.ceil(images.shape[0] / batch_size))
        for i in tqdm(range(n_batches)):
            low = int(i * batch_size)
            high = min(images.shape[0], int((i+1) * batch_size))
            cur_imgs = padded_imgs[low:high,:]
            cur_scat = scattering(cur_imgs).cpu()
            scat_features.append(cur_scat)
        scat_features = np.concatenate(scat_features, axis=0)
    else:
        # computing scattering all at once => fast but only possible for small datasets
        scat_features = scattering(padded_imgs)
        scat_features = np.array(scat_features.cpu(), dtype=np.float64)

    if(equalize):
        scat_features, _ = max_norm_equalization(scat_features)

    return scat_features


def convert_loader_to_scat(loader, scattering, device, equalize=False, verbose=0, pad_size=32):
    """
    Converting all images from a data loader to the scattering domain

    Args:
    -----
    loader: data loader
        Data loader fitting the images to convert to the scat domain
    scattering: kymatio network
        kymation scattering network used top process the images
    device: device
        cpu or gpu
    equalize: Boolean
        if True, scattering features are max-equalized
    verbose: integer
        verbosity level

    Returns:
    --------
    scat_features: numpy arrays
        scattering features from the images in the data loader
    """

    if(verbose > 0):
        iterator = tqdm(loader)
    else:
        iterator = loader

    scat_features = []
    imgs = []
    labels = []
    for i, (cur_imgs, cur_lbls) in enumerate(iterator):

        # preprocessing
        if(pad_size > 0):
            padded_imgs = pad_img(cur_imgs, target_shape=(pad_size,pad_size)).squeeze()
        padded_imgs = padded_imgs.to(device)

        # scattering forward
        cur_scat_features = scattering(padded_imgs)
        cur_scat_features = np.array(cur_scat_features.cpu(), dtype=np.float64)

        imgs.append(cur_imgs)
        scat_features.append(cur_scat_features)
        labels.append(cur_lbls)

    imgs = np.concatenate(imgs, axis=0)
    scat_features = np.concatenate(scat_features, axis=0)
    if(equalize):
        scat_features, _ = max_norm_equalization(scat_features)
    labels = np.concatenate(labels, axis=0)

    return imgs, scat_features, labels

#
