"""
Different methods for data visualization
"""

import numpy as np
from matplotlib import pyplot as plt


def display_subset_data(imgs, labels=None, shuffle=True, **kwargs):
    """
    Displaying a small subset of the images
    """

    fig, ax = plt.subplots(1, 6)
    fig.set_size_inches(30, 4)

    if(imgs.shape[1]<=3):
        imgs = imgs.transpose(0,2,3,1).squeeze()
    if(np.min(imgs) < 0):  # images are normalited to [-1, 1]
        imgs = (imgs + 1) / 2

    if(shuffle):
        indices = np.random.randint(low=0, high=imgs.shape[0], size=6)
    else:
        indices = np.arange(6)

    for col, idx in enumerate(indices):
        ax[col].imshow(imgs[idx,:])
        ax[col].axis("off")
        if(labels is not None):
            ax[col].set_title(f"Label {labels[idx]}")
    if("title" in kwargs):
        fig.suptitle(kwargs["title"])

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


###########################################################
# Visualizations of eigenvalues a histograms, CDFs and so on
#
# Patched-Imagenet/lib/visualizations
###########################################################

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def compute_eigenvalue_histogram(eigenvalues, title=None, bins=None, suptitle="", fig=None,
                                 ax=None, legend=None, savefig=None, figsize=None):
    """
    Computing plots that display the eigenvalue distribution as a histogram

    Args:
    -----
    eigenvalues: numpy array
        array with the eigenvalues of the covariance matrix sorted by magnitude in descending order
    label: integer or string
        label to include in the title of the plots
    suptitle: string
        subtitle of the figure
    """

    font = {'family': 'Times New Roman',
            'size' : 24}
    matplotlib.rc('font', **font)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    eigenvalues_norm = eigenvalues/np.sum(eigenvalues)

    if(fig is None or ax is None):
        fig, ax = plt.subplots(1,1)
        if(figsize is not None):
            fig.set_size_inches(*figsize)
        fig.suptitle(suptitle)

    if(bins is None):
        bins = np.linspace(0,max(eigenvalues_norm),60)

    if(legend is None):
        ax.hist(eigenvalues_norm, bins)
    else:
        ax.hist(eigenvalues_norm, bins, label=legend)
        ax.legend(loc="best")
    if(title is None):
        ax.set_title(f"Eigenvalue distribution")
    else:
        ax.set_title(title)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("# Eigenvalues")
    ax.set_yscale("log")
    plt.tight_layout()

    if(savefig is not None):
        plt.savefig(savefig)

    return fig, ax, bins


def compute_eigenvalue_cdf(eigenvalues, title=None, suptitle="", fig=None, ax=None, legend=None):
    """
    Computing CDF of the data variance as a function of the number of eigenvalues

    Args:
    -----
    eigenvalues: numpy array
        array with the eigenvalues of the covariance matrix sorted by magnitude in descending order
    label: integer or string
        label to include in the title of the plots
    suptitle: string
        subtitle of the figure
    """

    eigenvalues_norm = eigenvalues/np.sum(eigenvalues)
    cdf = np.cumsum(eigenvalues_norm)

    if(fig is None or ax is None):
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(12,5)
        fig.suptitle(suptitle)

    if(legend is None):
        ax.plot(cdf)
    else:
        ax.plot(cdf, label=legend)
        ax.legend(loc="best")

    if(title is None):
        ax.set_title("CDF of the Eigenvalue distribution")
    else:
        ax.set_title(title)
    ax.set_xlabel("Number of eigenvalues")
    ax.set_ylabel("Accumulated variance")

    return fig, ax


def display_principal_angles_histogram(angles, fig=None, ax=None, title="", legend=None,
                                       corrs=True, xlabel="", ylabel="", yscale="log",
                                       savefig=None):
    """
    Displaying a historgram with the principal angle distribution
    """

    font = {'family': 'Times New Roman',
            'size' : 18}
    matplotlib.rc('font', **font)
    plt.rcParams["font.family"] = "Times New Roman"

    if(fig is None):
        fig, ax = plt.subplots(1,1)
        # fig.set_size_inches(6,6)

    if(corrs):
        angles = np.cos(angles.tolist())
        bins = np.linspace(start=0, stop=1., num=50)
    else:
        bins = np.linspace(start=0, stop=1.6, num=40)

    if(legend is None):
        ax.hist(angles, bins)
    else:
        ax.hist(angles, bins, label=legend)
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)

    if(savefig is not None):
        plt.savefig(savefig)

    return fig, ax


def display_cluster_features(eigenvalues, prototype=None, disp_prot=False, **kwargs):
    """
    Displaying some plots with eigenvalue statistics and withe prototype if necessary

    Args:
    -----
    prototype: numpy array
        class prototype. Corresponds to the mean/median of the class
    eigenvalues: numpy array
        eigenvalues from the data matrix sorted in descending order
    disp_prot: boolean
        If True, displays the class prototype
    """

    eigenvalues_norm = eigenvalues/np.sum(eigenvalues)
    cdf = np.cumsum(eigenvalues_norm)

    if(disp_prot):
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(15,4)
    else:
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(11,4)

    bins = np.linspace(0,max(eigenvalues_norm),120)
    ax[0].hist(eigenvalues_norm, bins)
    ax[0].set_title("Histogram of Eigenvalues")
    ax[0].set_ylabel("Number of Eigenvalues")
    ax[0].set_xlabel("Magnitude of Eigenvalues")
    ax[0].set_yscale("log")

    ax[1].plot(cdf)
    ax[1].set_xlabel("Number of eigenvalues")
    ax[1].set_ylabel("Accumulated variance")
    ax[1].set_title("CDF of the Eigenvalue distribution")

    if(disp_prot):
        prototype = prototype.reshape(28, 28)
        ax[2].imshow(prototype)
        ax[2].set_title("Class Prototype")

    if("suptitle" in kwargs):
        plt.suptitle(kwargs["suptitle"])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return


def display_cluster_imgs(imgs, preds):
    """
    Displaying a subset of images from each cluster
    """

    n_clusters = len(np.unique(preds))
    for i in range(n_clusters):
        cur_idx = np.where(preds == i)
        cur_imgs = imgs[cur_idx, :][0,:]
        display_subset_data(cur_imgs, title=f"Images from Cluster {i}")

    return

#
