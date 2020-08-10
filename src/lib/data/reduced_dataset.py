"""
Classes and methods for processing reduced datasets (rMNIST)

@author: Angel Villar-Corrales
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from lib.data.data_processing import get_classwise_data


class ReducedDataset(Dataset):
    """
    Implementation of a reduced dataset for training a method using tiny number of samples

    Args:
    -----
    data: numpy array
        Array containing the images to fit the dataset. Shape is (n_imgs, img_shape)
    labels: numpy array
        Array with the labels for the images. Shaope is (n_imgs)
    n_imgs: integer
        number of images from each class to have in the reduced dataset
    """


    def __init__(self, data, labels, n_imgs=10):
        """
        Initializing the reduced dataset object
        """

        if(torch.is_tensor(data)):
            data = data.numpy()
        if(len(data.shape)==3):
            data = data[:,np.newaxis,:,:]
        self.all_data = data
        self.all_labels = labels
        self.n_imgs = n_imgs
        self.data, self.labels = self.sample_reduced_data()

        return


    def __len__(self):
        """
        Obtainign number of samples in the dataset
        """

        n_samples = len(self.labels)
        return n_samples


    def __getitem__(self, idx):
        """
        Sampling the pair (img, target) for the idx element of the dataset
        """

        img = self.data[idx]
        target = self.labels[idx]
        return (img, target)


    def sample_reduced_data(self, shuffle=True):
        """
        Sampling a new reduced dataset from the complete data

        Args:
        -----
        shuffle: boolean
            If true, data and labels are randomly shuffled. otherwise they come sorted

        Returns:
        --------
        r_data, r_labels: numpy array
            arrays containing both the reduced dataset images and labels
        """

        labels_unique = np.unique(self.all_labels)
        num_classes = len(labels_unique)

        r_data = np.empty((0, *self.all_data[0,:].shape))
        r_labels =  np.array([])

        # samplig n_dims from each class
        for l in labels_unique:
            class_data = get_classwise_data(data=self.all_data,
                                            labels=self.all_labels,
                                            label=l)
            idx = np.random.randint(low=0, high=len(class_data), size=self.n_imgs)
            class_data = class_data[idx,:]
            r_data = np.concatenate((r_data, class_data), axis=0)
            r_labels = np.append(r_labels, [l]*self.n_imgs)

        if(shuffle):
            p = np.random.permutation(len(r_labels))
            r_data, r_labels = r_data[p,:], r_labels[p]

        return r_data, r_labels


#
