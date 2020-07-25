###########################################################
# Methods to compute evaluation metrics given certain parameteres
# Patched-Imagenet/lib/utils
###########################################################

import numpy as np
import torch


def compute_loss(outputs, labels, loss_function, patch_processing="channelwise"):
    """
    Computing the loss values given the original and predicted labels, and the loss function

    Args:
    -----
    outputs: torch tensor
        activation of the neuros in the output layer of the neural network
    labels: torch tensor
        labels corresponding to the images that have been fed to the network
    loss_function: torch.nn function
        loss function used to compute the loss value
    patch_processing: string
        Method in which the different patches are treated: ['channelwise', 'batchwise']

    Returns:
    --------
    loss: float
        loss values computed using input parameters
    """

    if(patch_processing == "channelwise"):
         loss = loss_function(input=outputs, target=labels)

    elif(patch_processing == "batchwise"):
        labels = labels.repeat(outputs.shape[0])
        loss = loss_function(input=outputs, target=labels)

    return loss


def compute_accuracy(outputs, labels, label_list, patch_processing="channelwise"):
    """
    Computing the accuracy given the original labels and the ones predicted by the network

    Args:
    -----
    outputs: torch tensor
        activation of the neuros in the output layer of the neural network
    labels: torch tensor
        labels corresponding to the images that have been fed to the network
    label_list: list
        list with the different labels for the dataset being used
        i.e., for MNIST : ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
              for two_class: ['dog', 'rabbit']
    patch_processing: string
        Method in which the different patches are treated: ['channelwise', 'batchwise']

    Returns:
    --------
    n_correct_labels: integer
        number of labels in the batch that have been correctly predicted
    """

    # moving variables to cpu memory and traing outputs as np array
    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu()

    # when patches are processed channelwise, then the predicted label simply corresponds
    # to the highest activated output neuron
    if(patch_processing == "channelwise"):
        predicted_labels = label_list[np.argmax(outputs, axis=1)]
        n_correct_labels =  len(np.where(predicted_labels == labels)[0])

    # when patches are processed patchwise, then we apply the BoSF approach: a classwise
    # activation is computed for each patch. Then, all activation contributions from the same
    # class are summed. Then a softmax is applied and the predicted label will be the one
    # corresponding to the class with the highest likelihood
    elif(patch_processing == "batchwise"):
        summed_activations = np.sum(outputs, axis=0)
        # summed_activations = np.mean(outputs, axis=0)
        exponentials = np.exp(summed_activations)
        likelihoods = exponentials/np.sum(exponentials)
        predicted_labels = label_list[np.argmax(likelihoods)]
        n_correct_labels =  len(np.where(predicted_labels == labels[0])[0])

    return n_correct_labels



def compute_pairwise_distances(scat_coeffs, verbose=False, norm=""):
    """
    Computing the euclidean distance between each pair of scattering coefficients

    Args:
    -----
    scat_coeffs: numpy array (N,...)
        numpy array with number of images/patches as first dimension
    verbose: boolean
        verbosity activated or deactivated
    norm: row/col/matrix/none
        type of normalization to be applied
    distance_matrix: numpy array (N,N)
        matrix with the distance between each pair of elements in each position
    """

    distance_matrix = np.zeros((scat_coeffs.shape[0],scat_coeffs.shape[0]))

    for i in range(scat_coeffs.shape[0]):
        for j in range(scat_coeffs.shape[0]):
            if(i==j):
                continue
            distance = np.linalg.norm( (scat_coeffs[i,:]-scat_coeffs[j,:]).flatten(), ord=2)
            distance_matrix[i,j] = distance

            if(verbose):
                print(f"Distance between patches {i+1} and {j+1} is {distance}")

        if(verbose):
            print("\n\n")

    if(norm=="row"):
        for i in range(distance_matrix.shape[0]):
            distance_matrix[i,:] = distance_matrix[i,:]/np.max(distance_matrix[i,:])
    elif(norm=="col"):
        for i in range(distance_matrix.shape[0]):
            distance_matrix[:,i] = distance_matrix[:,i]/np.max(distance_matrix[:,i])
    elif(norm=="matrix"):
        distance_matrix = distance_matrix/np.max(distance_matrix)

    return distance_matrix
