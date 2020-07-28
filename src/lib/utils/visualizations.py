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


#
