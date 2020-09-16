"""
Methods for loading and preprocessing the datasets

@author: Angel Villar-Corrales
"""

import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from CONFIG import CONFIG


class ClassificationDataset(Dataset):
    """
    Class used for handling the different datasets for classification

    Args:
    -----
    data_path: string
        path from where the data will be stored or otherwise downloaded
    dataset_name: string
        dataset to load ['mnist', 'svhn']
    valid_size: float
        percentage of the data used for validaton. Must be in range [0,1)
    transformations: list
        list containing the augmentations and preprocessing methods to apply
    shuffle: boolean
        if True, train/valid split is done randmly
    """

    def __init__(self, data_path, dataset_name="mnist", valid_size=0.2,
                 transformations=None, shuffle=False):
        """
        Initializer of the classification dataset object
        """

        # checking valid values for the parameters
        assert dataset_name in ["mnist", "svhn", "fashion_mnist", 'usps', 'mnist-test', \
               'coil-100'], f"Dataset name: {dataset_name} is not a correct value. " \
                f"Choose one from ['mnist', 'svhn', 'usps', 'mnist-test', 'coil-100']"
        assert (valid_size >= 0 and valid_size < 1), f"Valid size must be in range [0,1)"

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.valid_size = valid_size
        self.shuffle = shuffle

        # enforcing ToTensor in the transforms
        if(transformations is None):
            transformations = []
        if(transforms.ToTensor() not in transformations):
            transformations.append(transforms.ToTensor())
        transformations.append(transforms.Normalize((0.5,), (0.5,)))
        transformations = transforms.Compose(transformations)

        # loading the corresponding data
        if(dataset_name == "mnist"):
            train_set = datasets.MNIST(self.data_path, train=True, download=True,
                                            transform=transformations)
            test_set = datasets.MNIST(self.data_path, train=False, download=True,
                                        transform=transformations)

        elif(dataset_name == "mnist-test"):
            train_set = None
            test_set = datasets.MNIST(self.data_path, train=False, download=True,
                                        transform=transformations)

        elif(dataset_name == "svhn"):
            train_set = datasets.SVHN(self.data_path, split='train',download=True,
                                      transform=transformations)
            test_set = datasets.SVHN(self.data_path, split='test',download=True,
                                      transform=transformations)
            train_set.targets, test_set.targets = train_set.labels, test_set.labels

        elif(dataset_name == "fashion_mnist"):
            train_set = datasets.FashionMNIST(self.data_path, train=True, download=True,
                                              transform=transformations)
            test_set = datasets.FashionMNIST(self.data_path, train=False, download=True,
                                             transform=transformations)

        elif(dataset_name == "usps"):
            train_set = datasets.USPS(self.data_path, train=True, download=True,
                                      transform=transformations)
            test_set = datasets.USPS(self.data_path, train=False, download=True,
                                     transform=transformations)

        elif(dataset_name == "coil-100"):
            data_path = os.path.join(self.data_path, "coil-100", "coil-100")
            get_lbl = lambda name: int(name.split("_")[0][3:])
            train_set = None
            test_set = CustomDataset(root=data_path,
                                     transform=transformations,
                                     get_lbl=get_lbl)

        if(train_set is not None):
            self.train_data, self.train_labels = train_set.data, train_set.targets
        self.test_data, self.test_labels = test_set.data, test_set.targets
        self.train_set = train_set
        self.test_set = test_set
        if(self.valid_size > 0 and self.train_set is not None):
            self._get_train_validation_split()

        return


    def _get_train_validation_split(self, valid_size=None):
        """
        Splitting the training data into train/validation sets
        """

        # updating valid size if necessary
        if(valid_size is not None):
            assert (valid_size >= 0 and valid_size < 1), f"Valid size must be in range [0,1)"
            self.valid_size = valid_size

        num_train = len(self.train_set)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        # randomizing train and validation set
        if(self.shuffle):
            np.random.seed(CONFIG["random_seed"])
            np.random.shuffle(indices)

        # getting idx for train and validation
        train_idx, val_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(val_idx)

        self.train_data, self.train_labels = self.train_data[train_idx], self.train_labels[train_idx]
        self.valid_data, self.valid_labels = self.train_data[val_idx], self.train_labels[val_idx]

        return


    def get_data_loader(self, split="train", batch_size=128, shuffle=False):
        """
        Obtaining a data loader for a certain dataset split

        Args:
        -----
        split: string
            dataset split to fit to the dataloadr. ['train', 'valid', 'test']
        batch_size: integer
            number of example in a minibathc
        shuffle: boolean
            if true, batches are drawn randomly

        Returns:
        --------
        loader: DataLoader
            Dataloader object fitting the corresponding dataset split
        """

        assert split in ["train", "valid", "test"]
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(shuffle, bool)

        num_workers = CONFIG["num_workers"]
        if(split is "train"):
            dataset_split = self.train_set
            sampler = self.train_sampler
        elif(split is "valid"):
            dataset_split = self.train_set
            sampler = self.valid_sampler
        else:
            dataset_split = self.test_set
            sampler = None

        loader = torch.utils.data.DataLoader(dataset_split, batch_size=batch_size,
                                             num_workers=num_workers, sampler=sampler)

        return loader


    def get_all_data(self):
        """
        Obtaining all images and labels in the dataset
        """

        # for MNIST-test we just get the test set
        if(self.train_set is None):
            return self.test_set.data.numpy(), self.test_set.targets.numpy()

        # for other datasets we concatenate all data
        train_data, train_labels = self.train_set.data, self.train_set.targets
        test_data, test_labels = self.test_set.data, self.test_set.targets

        all_data = np.concatenate((train_data, test_data), axis=0)
        all_labels = np.concatenate((train_labels, test_labels), axis=0)

        return all_data, all_labels


class CustomDataset(Dataset):
    def __init__(self, root, transform, get_lbl=None):
        self.root = root
        self.transform = transform
        self.get_lbl = get_lbl
        self.data, self.targets = None, None
        self._load_data()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx,:], self.targets[idx]

    def _load_data(self):
        data, targets = [], []
        for img_name in os.listdir(self.root):
            if(img_name[-4:] != ".png"):
                continue
            label = self.get_lbl(img_name)
            img = os.path.join(self.root, img_name)
            img = np.array(Image.open(img).convert('L'))
            img = self.transform(img)
            data.append(img)
            targets.append(label)
        self.data = torch.stack(data)
        self.targets = torch.Tensor(targets)

#
