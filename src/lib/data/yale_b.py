"""
Class for loading and processing the YaleB dataset:
    http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html
This dataset contains cropped faces of 38 people under 64 differenc lightning conditions
We use this dataset to test the subspace clustering algorithms
"""

import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class YaleB(Dataset):
    """
    Custom implementation of the Dataset for handling the Yale_B face database
    """

    def __init__(self, data_path=None, shuffle=False):
        """
        Initializer of the YaleB object
        """

        self.data_path = data_path
        self.shuffle = shuffle

        self.data = []
        self.labels = []
        self.image_paths = []

        self._get_img_paths()

        return


    def __len__(self):
        """
        Getting the number of elements in the dataset
        """

        length = len(labels)
        return length


    def __getitem__(self, idx, downsample=False, flatten=False):
        """
        Extracting an image from the dataset
        """

        cur_path = self.image_paths[idx]
        img = np.array(Image.open(cur_path))
        if(downsample):
            img = img[::4, ::4]
        if(flatten):
            img = img.flatten()
        label = int(cur_path.split("/")[-1][5:7])

        return img, label


    def get_data_matrix(self, L=0, downsample=False, flatten=True, shuffle=False,
                        standarize=False, verbose=0, random_seed=13):
        """
        Obtaining the data as a (m, N) matrix

        Args:
        -----
        L: integer
            number of different classes to consider. 0 means all classes
        downsample: boolean
            If True, images are donwsampled from (192×168) to (48 × 42)
        flatten: boolean
            If True, images are flattened into a vector representation
        shuffle: boolean
            If true, paths to images are randomly shuffled before sampling
        standarize: boolean
            If True, data is standarized to have zero-mean and unit variance
        verbose: integer
            verbosity level
        """

        data_matrix = None
        data_list = []
        labels = []
        classes = []

        cur_paths = self.image_paths
        if(shuffle):
            np.random.seed(random_seed)
            np.random.shuffle(cur_paths)
        else:
            cur_paths = sorted(cur_paths)

        for path in cur_paths:

            # loading label and checking if image will be loaded
            cur_label = int(path.split("/")[-1][5:7])
            if(L>0 and len(classes)==L and cur_label not in classes):
                continue
            if(cur_label not in classes):
                classes.append(cur_label)

            # loading image and preprocessing
            cur_img = np.array(Image.open(path))
            if(downsample):
                cur_img = cur_img[::4, ::4]
            if(flatten):
                cur_img = cur_img.flatten()
            data_list.append(cur_img)
            labels.append(cur_label)

        data_matrix = np.array(data_list)
        if(flatten):
            data_matrix = data_matrix.T
        labels = np.array(labels)

        if(standarize):
            data_matrix = StandardScaler().fit_transform(data_matrix)

        if(verbose > 0):
            n_imgs = len(labels)
            print(f"Loading {n_imgs} of shape {data_matrix[0,:].shape}")
            print(f"Images belong to one out of {L} classes: {np.unique(labels)}")

        return data_matrix, labels


    def _get_img_paths(self):
        """
        Saving image paths in a list for online loading
        """

        if(self.data_path is None):
            root_path = os.getcwd()
            self.data_path = os.path.join(root_path, "data", "CroppedYale")

        # obtaining image paths
        images = []
        for filename in Path(self.data_path).rglob("*/*"):
            filename = str(filename)
            if("Ambient" in filename or ".pgm" not in filename):
                continue
            images.append(filename)
        if(len(images)==0):
            print(f"WARNING! No images have been found in data directory {self.data_path}")

        # randomly shuffling if necessary
        if(self.shuffle):
            np.random.shuffle(images)
        else:
            images = sorted(images)

        self.image_paths = images

        return


#
