"""
Useful methods for several purposes
Patched-Imagenet/lib/utils

@author: Angel Villar-Corrales
"""

import os
import json
import datetime

import numpy as np

import torch


def timestamp():
    """
    Computes and returns current timestamp

    Args:
    -----
    timestamp: String
        Current timestamp in formate: hh-mm-ss
    """

    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp


def for_all_methods(decorator):
    """
    Decorator that applies a decorator to all methods inside a class
    """
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def create_directory(path):
    """
    Method that creates a directory if it does not already exist

    Args
    ----
    path: String
        path to the directory to be created
    dir_existed: boolean
        Flag that captures of the directory existed or was created
    """
    dir_existed = True
    if not os.path.exists(path):
        os.makedirs(path)
        dir_existed = False
    return dir_existed


def get_loss_stats(loss_list):
    """
    Computes loss statistics given a list of loss values

    Args:
    -----
    loss_list: List
        List containing several loss values
    """

    if(len(loss_list)==0):
        return

    loss_np = torch.stack(loss_list)
    avg_loss = torch.mean(loss_np)
    max_loss = torch.max(loss_np)
    min_loss = torch.min(loss_np)
    print(f"Average loss: {avg_loss} -- Min Loss: {min_loss} -- Max Loss: {max_loss}")

    return avg_loss


#####################################
# Utils for creating a new experiment:
#   - creating experiment info json file
#   - saving network architecture to .txt
#   - Updating info and retrieving info
#####################################


def create_experiment(output_path, output_dir, dataset, model, valid_size, learning_rate,
                      batch_size, early_stopping, max_epochs, bottleneck_dim=None, num_patches=None,
                      debug=True, num_layers=None, equalization="", processing="batchwise",
                      sampling_mode="equal", activation="relu", **kwargs):
    """
    Creates a json file with metadata about the experiment

    Args:
    -----
    output_name: String
        Path to the experiment folder
    output_dir: String
        Folder where the outputs experiments will be saved
    dataset: String
        name of the dataset the model has been trained on
    model_type: String
        name of the model architecture
    valid_size: float [0-1]
        amount of corruption added to the training labels
    learning_rate: Float
        learning rate
    batch_size: Int
        batch size
    early_stopping: Boolean
        Flag whether the early stopping condition will be applied or not
    max_epochs: Int
        Maximum nuber of epochs to be executed
    bottleneck_dim: Int
        Dimension of the hidden fully connected layer
    num_patches: Integer
        Number of patches used for every image in the batch
    num_layers: Integer
        number of hidden layers to use in the fully connected frontend
    equalization: String
        Type of scattering coefficient equalization to be applied [batch-normalization, train-normalization, none]
    processing: String
        Method used for processing the patches: channelwise or batchwise
    debug: Boolean
        Flag that selects debug or production mode
    """

    filepath = os.path.join(output_path, "experiment_data.json")

    # if it already exists, we do not overwrite it
    if(os.path.exists(filepath)):
        return filepath

    data = {}
    data["experiment_started"] = timestamp()
    data["dataset"] = dataset
    data["learning_rate"] = learning_rate
    data["batch_size"] = batch_size
    data["valid_size"] = valid_size
    data["early_stopping"] = early_stopping
    data["max_epochs"] = max_epochs
    data["num_patches"] = num_patches
    data["equalization"] = equalization
    data["patch_processing"] = processing
    data["sampling_mode"] = sampling_mode
    data["debug_mode"] = debug

    # model information
    data["model"] = {}
    data["model"]["model_name"] = model
    data["model"]["bottleneck_dim"] = bottleneck_dim
    data["model"]["num_layers"] = num_layers
    data["model"]["activation"] = activation

    # loss and accuracy information
    data["loss"] = {}
    data["loss"]["train_loss"] = []
    data["loss"]["valid_loss"] = []
    data["accuracy"] = {}
    data["accuracy"]["train_accuracy"] = []
    data["accuracy"]["valid_accuracy"] = []


    with open(filepath, "w") as file:
        json.dump(data, file)

    return filepath


def save_network_architecture(filepath, architecture, loss_function, optimizer, scattering_architecture=""):
    """
    Saves the network architecture (layers, params, optimizer, ...) into the metadata file
    Creates an extra file to dump the architecture in a human-readable way
    """

    # saving architecture, loss and optimizer in the metadata file
    with open(filepath) as filedata:
        data = json.load(filedata)

    data["model"]["model_architecture"] = architecture
    data["model"]["scattering_architecture"] = scattering_architecture
    data["optimizer"] = optimizer.__class__.__name__
    data["loss_function"] = loss_function.__class__.__name__

    with open(filepath, "w") as file:
        json.dump(data, file)

    # creating file and saving architecture in a human-readable way
    directory = "/".join(filepath.split("/")[:-1])
    new_file = os.path.join(directory, "network_architecture.txt")

    f = open(new_file, "w")
    f.write(architecture)
    f.write("\n\n\n")
    f.write(scattering_architecture)
    f.close()

    return


def add_information_to_experiment(filepath, train_loss=None, valid_loss=None, train_accuracy=None,
                                  valid_accuracy=None):
    """
    Adds information to the JSON data
    """

    with open(filepath) as filedata:
        data = json.load(filedata)

    data["loss"]["train_loss"].append(train_loss)
    data["loss"]["valid_loss"].append(valid_loss)
    data["accuracy"]["train_accuracy"].append(train_accuracy)
    data["accuracy"]["valid_accuracy"].append(valid_accuracy)

    with open(filepath, "w") as file:
        json.dump(data, file)

    return


def add_test_accuracy(filepath, test_accuracy):
    """
    Adding accurac on the test set to the experiment datafile
    """

    with open(filepath) as filedata:
        data = json.load(filedata)

    data["accuracy"]["test_accuracy"] = test_accuracy

    with open(filepath, "w") as file:
        json.dump(data, file)

    return

def add_list_test_accuracy(filepath, test_epochs, test_accuracy):
    """
    Adding accurac on the test set to the experiment datafile
    """

    with open(filepath) as filedata:
        data = json.load(filedata)

    data["accuracy"]["list_test_accuracy"] = {}
    data["accuracy"]["list_test_accuracy"]["epochs"] = test_epochs
    data["accuracy"]["list_test_accuracy"]["test_accuracy"] = test_accuracy

    with open(filepath, "w") as file:
        json.dump(data, file)

    return


def get_values(dictionary, keys, metric="train_accuracy"):
    """
    Returns a vector with the given metric collected from the dictionary
    """

    values = []
    for key in keys:
        values.append(dictionary[key][metric])
    return values


#####################################
# Utils for retrieval experiment:
#   - creating retrieval info json file
#   - loading previous info
#####################################


def load_retrieval_data(experiment_dir, dims_method="pca"):
    """
    Loading the data from a previously retrieval experiment

    Args:
    -----
    experiment_dir: string
        path to the experiment directory
    dims_method: string
        Method used for dimensionality reduction ['pca', 'poc']

    Returns:
    --------
    data: json
        json containing the retrieval data
    """

    filepath = os.path.join(experiment_dir, f"retrieval_data_{dims_method}.json")

    if(not os.path.exists(filepath)):
        data = create_retrieval_experiment(experiment_dir, dims_method=dims_method)
    else:
        with open(filepath) as filedata:
            data = json.load(filedata)

    return data


def create_retrieval_experiment(experiment_dir, dims_method="pca"):
    """
    Creating a new json file for a neighbor retrieval experiment

    Args:
    -----
    experiment_dir: string
        path to the experiment directory
    dims_method: string
        Method used for dimensionality reduction ['pca', 'poc']

    Returns:
    --------
    data: json
        json containing the retrieval data
    """

    filepath = os.path.join(experiment_dir, f"retrieval_data_{dims_method}.json")

    data = {}
    data["last_modified"] = timestamp()

    data["scattering"] = {}
    data["scattering"]["tree"] = {}
    data["scattering"]["tree"]["datapoints_used"] = ""
    data["scattering"]["tree"]["neighbors_retrieved"] = ""
    data["scattering"]["tree"]["duration"] = ""
    data["scattering"]["tree"]["precision"] = ""
    data["scattering"]["tree"]["distribution"] = {}
    data["scattering"]["graph"] = {}
    data["scattering"]["graph"]["datapoints_used"] = ""
    data["scattering"]["graph"]["neighbors_retrieved"] = ""
    data["scattering"]["graph"]["duration"] = ""
    data["scattering"]["graph"]["precision"] = ""
    data["scattering"]["graph"]["distribution"] = {}

    data["raw"] = {}
    data["raw"]["tree"] = {}
    data["raw"]["tree"]["datapoints_used"] = ""
    data["raw"]["tree"]["neighbors_retrieved"] = ""
    data["raw"]["tree"]["duration"] = ""
    data["raw"]["tree"]["precision"] = ""
    data["raw"]["tree"]["distribution"] = {}
    data["raw"]["graph"] = {}
    data["raw"]["graph"]["datapoints_used"] = ""
    data["raw"]["graph"]["neighbors_retrieved"] = ""
    data["raw"]["graph"]["duration"] = ""
    data["raw"]["graph"]["precision"] = ""
    data["raw"]["graph"]["distribution"] = {}

    with open(filepath, "w") as file:
        json.dump(data, file)

    return data



#####################################
# Utils for classification experiment:
#   - creating retrieval info json file
#   - loading previous info
#####################################


def load_classification_data(experiment_dir):
    """
    Loading the data from a previously classification experiment

    Args:
    -----
    experiment_dir: string
        path to the experiment directory
    data: json
        json containing the classification data
    """

    filepath = os.path.join(experiment_dir, "classification_data.json")

    if(not os.path.exists(filepath)):
        data =create_classification_experiment(experiment_dir)
    else:
        with open(filepath) as filedata:
            data = json.load(filedata)

    return data


def create_classification_experiment(experiment_dir):
    """
    Creating a new json file for a cluster classification experiment

    Args:
    -----
    experiment_dir: string
        path to the experiment directory
    data: json
        json containing the classification data
    """

    filepath = os.path.join(experiment_dir, "classification_data.json")

    data = {}
    data["last_modified"] = timestamp()

    with open(filepath, "w") as file:
        json.dump(data, file)

    return data


#####################################
# Utils for principal component analysis:
#   - creating principal components info json file
#   - loading previous info file
#####################################


def load_principal_components_data(experiment_dir, type="standard_data"):
    """
    Loading the data from a previously principal component analysis

    Args:
    -----
    experiment_dir: string
        path to the experiment directory
    data: json
        json containing the principal component data
    """

    if(type=="standard_data"):
        filepath = os.path.join(experiment_dir, "principal_component_analysis.json")
    else:
        filepath = os.path.join(experiment_dir, "principal_component_analysis_normalized.json")

    if(not os.path.exists(filepath)):
        data =create_principal_componenet_experiment(experiment_dir)
    else:
        with open(filepath) as filedata:
            data = json.load(filedata)

    return data


def create_principal_componenet_experiment(experiment_dir):
    """
    Creating a new json file for a principal component analysis

    Args:
    -----
    experiment_dir: string
        path to the experiment directory
    data: json
        json containing the principal component data
    """

    filepath = os.path.join(experiment_dir, "principal_component_analysis.json")

    data = {}
    data["last_modified"] = timestamp()

    with open(filepath, "w") as file:
        json.dump(data, file)

    return data


def save_grid_images(images, outputs, labels, name):
    """
    Saving a grid of 6 images displaying inputs with their original and predicted labels

    Args:
    -----
    images: torch tensor
        input image fed to the networks
    outputs: torch tensor
        predicitions of the neural network. Correspond to the likelihoods for each class
    labels: numpy array
        original labels corresponding to the images
    name: string
        name used tos ave the image
    """

    outputs = outputs.cpu().detach().numpy()

    image = images[0:6].cpu().numpy()
    image = np.transpose(image,(0,2,3,1))
    output = outputs[0:6]
    idx = np.argmax(output,axis=1)

    fig,ax = plt.subplots(2,3)
    for i in range(6):
        row = i//3
        col = i%3
        ax[row,col].imshow(image[i,:,:,0])
        ax[row,col].set_title(f"Predicted: {idx[i]}; real: {labels[i]}")

    img_path = os.path.join(os.getcwd(),"outputs","img")
    dir_existed = utils.create_directory(img_path)
    plt.savefig( os.path.join(img_path, name))

#
