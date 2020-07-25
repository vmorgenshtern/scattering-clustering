###########################################################
# Auxiliary methods used to write and read efficiently numpy
# .npz compressed files
# Patched-Imagenet/lib/utils
###########################################################

import os
import json
import random
from tqdm import tqdm

import numpy as np

from lib.data_loader.data_loader import Dataset


def save_npz(file, value):
    """
    Saving  a given value in a .npz file

    Args:
    -----
    file: string
        path to the .npz file where the values are stored
    value: dictionary
        dictionary containing pairs of names and values
    """

    np.savez(file, **value)

    return


def load_npz(file):
    """
    Loading a previously computed .npz file

    Args:
    -----
    file: string
        path to the .npz file where the values are stored
    loader: Object
        Loader object used to fetch the stored variables
    """

    if(not os.path.exists(file)):
        print(f"ERROR\nFile {file} does not exist")
        assert False

    loader = np.load(file)
    return loader


def load_scattering_features(feature_path, num_files_to_load=-1, shuffle=False, flatten=True, num_patches=100):
    """
    Method that efficiently loads the scattering features from the npz files

    Args:
    -----
    feature_path: string
        path where the npz files are stores
    num_files_to_load: integer
        maximum number of files to load. If -1, all files are loaded
    shuffle: Boolean
        Flag that chooses whether files are read in correct order or in random order
    flatten: Boolean
        Flag that decides whether scattering features are flattenend or not
        (B, X*Y*Z) or (B,X,Y,Z)
    num_patches: Integer
        Number of patches taken from an image
    """

    list_files = os.listdir(feature_path)
    if(shuffle):
        random.shuffle(list_files)
    num_files = len(list_files)

    features = []
    labels = []

    # loading features and label for every image
    for i,file in enumerate(tqdm(list_files)):

        if(file[-4:]!=".npz"):
            continue

        data_path = os.path.join(feature_path, file)
        loader = load_npz(data_path)
        features.append(loader['features'])

        if(len(loader["features"].shape)==5):
            for i in range(loader["features"].shape[0]//num_patches):
                labels += [loader["labels"][i]]*num_patches
                # stopping condition
                if(len(labels)>num_files_to_load and num_files_to_load>0):
                    break
        else:
            labels += loader["labels"].tolist()

        # stopping condition
        if(len(labels)>num_files_to_load and num_files_to_load>0):
            break

    features = np.concatenate(features, axis=0)

    if(num_files_to_load>0):
        boundary = min(features.shape[0], num_files_to_load)
    else:
        boundary = features.shape[0]

    if(flatten):
        if(len(features.shape)==5):
            features = features[:boundary,:,:,:,:]
            features = np.array([features[i,:,:,:,:].flatten() for i in range(features.shape[0])])
            labels = labels[:boundary]
        elif(len(features.shape)==4):
            features = features[:boundary,:,:,:]
            features = np.array([features[i,:,:,:].flatten() for i in range(features.shape[0])])
            labels = labels[:boundary]
        else:
            features = features[:boundary,:]
            features = np.squeeze(features)
            labels = labels[:boundary]

    labels = np.array(labels)

    return features, labels



def load_data(feature_type, exp_directory, num_files=-1, split="train", flatten=True):
    """
    Loading scattering features or raw features depending on the given feature type

    Args:
    -----
    feature_type: string
        type of feautre being analyzed (raw or scattering)
    exp_directory: string
        path to the directory where feaures are stored and where outputs will be saved
    num_files: integer
        number of files to average to compute the principal components
    split: string [training or test]
        split of the dataset to compute the principal components of
    flatten: boolean
        decided whether flattening the scattering feaures
    """

    if(feature_type == "raw"):
        feature, labels = load_dataset(num_files, split, exp_directory)
    elif(feature_type == "scattering" or feature_type == "cleaned_scattering"):
        feature, labels = get_scattering_features(exp_directory, num_files, split, feature_type, flatten=flatten)
    else:
        print(f"Error! Feature type {feature_type} is not recognized. Use ['raw', 'scattering']...")
        exit()

    return feature, labels


def load_dataset(num_files, split, exp_directory):
    """
    Loading dataset using a torch data loader

    Args:
    -----
    num_files: integer
        number of files to average to compute the principal components
    split: string [training or test]
        split of the dataset to compute the principal components of
    features: list of numpy arrays
        list containing the scattering features of every example
    labels: list of integers
        list containing the label of every example
    """

    # getting the dataset used in the experiment
    experiment_file = os.path.join(exp_directory, "experiment_data.json")
    with open(experiment_file)  as filedata:
        experiment_data = json.load(filedata)
    dataset = experiment_data["dataset"]

    # loading the test set
    root_path = os.getcwd()
    data_path = os.path.join(root_path, "data")
    dataset = Dataset(data_path=data_path, use_gpu=True, dataset=dataset, num_patches=100,
                      debug=False, batch_size=1, shuffle=True, experiment_path=exp_directory)

    if(split == "test"):
        loader = dataset.get_test_set()
        set = dataset.test_set
    else:
        loader = dataset.get_combined_loader()
        set = dataset.train_set

    print(f"Set contains {len(set)} images")
    print(f"Using a maximum of {num_files} images")

    features = []
    labels = []
    for i, point in enumerate(set):
        aux = point[0]
        for j in range(aux.shape[0]):
            features.append(aux[j,:,:].numpy().flatten())
            labels.append(point[1])
        if(len(features)>num_files and num_files!=-1):
            break

    features = features[:num_files]
    labels = labels[:num_files]

    return features, labels


def get_scattering_features(exp_directory, num_files_to_average, split, feature_type, flatten=True):
    """
    Loading scattering features and labels

    Args:
    -----
    num_files_to_average: integer
        number of files to average to compute the principal components
    exp_directory: string
        path to the experiment folder
    split: string [training or test]
        split of the dataset to compute the principal components of
    feature_type: string
        Type of scattering features to be loaded: regular or cleaned
    features: list of numpy arrays
        list containing the scattering features of every example
    labels: list of integers
        list containing the label of every example
    """

    features = []
    labels = []

    if(feature_type == "scattering"):
        folder = "scattering_features"
    else:
        folder = "cleaned_scattering_features"

    feature_directory = os.path.join(exp_directory, folder, f"scattering_features_{split}")
    features, labels = load_scattering_features(feature_directory, num_files_to_load=num_files_to_average,
                                                flatten=flatten, shuffle=True)

    return features, labels



#
