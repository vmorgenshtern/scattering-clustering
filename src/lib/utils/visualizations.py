"""
Different methods for data visualization

@author: Angel Villar-Corrales
"""

import numpy as np
from matplotlib import pyplot as plt


def display_subset_data(imgs, labels):
    """
    Displaying a small subset of the images
    """

    fig, ax = plt.subplots(1, 6)
    fig.set_size_inches(30, 4)

    for col in range(6):
        ax[col].imshow(imgs[col,:])
        ax[col].axis("off")
        ax[col].set_title(f"Label {labels[col]}")

    return


def visualize_accuracy_landscape(xaxis, accuracy, **kwargs):
    """
    Visualizing a line plot displaying the accuracy
    """

    if("ax" in kwargs):
        ax = kwargs["ax"]
    else:
        fig, ax = plt.subplots(1,1)

    ax.scatter(xaxis, accuracy, linewidth=3)
    ax.plot(xaxis, accuracy, linewidth=3)

    if("grid" in kwargs):
        ax.grid()
    if("title" in kwargs):
        ax.set_title(kwargs["title"])
    if("xlabel" in kwargs):
        ax.set_xlabel(kwargs["xlabel"])
    if("ylabel" in kwargs):
        ax.set_ylabel(kwargs["ylabel"])
    if("xticks" in kwargs):
        xticklabels = kwargs["xticks"]
        xticks = np.arange(len(xticklabels))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    if("yticks" in kwargs):
        yticklabels = kwargs["yticks"]
        yticks = np.arange(len(yticklabels))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    return

#
