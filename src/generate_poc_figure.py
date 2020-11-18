"""
Generating a figure for the paper illustrating the POC algorithm


@author: Angel Villar-Corrales
"""

import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from lib.projections.POC import POC


def display_space(points, labels, n_dim=2, n_points=-1, title="", fig=None,
                  ax=None, colors=None, alpha=None, **kwargs):
    """
    Displaying the points in the 1, 2 or 3 dimensional space and coloring them given their label

    Args:
    ----
    points: np array
        data points to display
    labels: np array
        array with the labels of the data points
    n_dim: integer
        dimensionality of the space to visualize {1,2,3}
    n_points: integer
        number of points to visualize. If -1, all points are plotted
    legend: list
        labels for each class to display in the legend.
        If None, .If 'class' class number is used.
    """

    font = {'family': 'Times New Roman',
            'size' : 26}
    matplotlib.rc('font', **font)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


    # figure parameters and useful variables
    n_classes = len(set(labels))
    cb_labels = [str(i) for i in range(n_classes)]
    if(colors is None):
        colors = ['r', 'b', 'g', 'y', 'purple', 'orange', 'k', 'brown', 'grey', 'c'][:n_classes]
    if(alpha is None):
        alpha = [1]*n_classes

    if(fig is None or ax is None):
        fig = plt.figure(figsize=(10,8))
        if(n_dim<3):
            ax = plt.axes()
        elif(n_dim==3):
            ax = plt.axes(projection="3d")

    ax.set_title(title)

    display_legend = True
    if("legend" not in kwargs):
        display_legend = False
        legend = [f"class {l}" for l in np.unique(labels)]
    elif(len(kwargs["legend"])==0):
        legend = [f"class {l}" for l in np.unique(labels)]

    # creating and saving figure
    for i,l in enumerate(np.unique(labels)):
        idx = np.where(l==labels)
        if(n_dim==1):
            zeros = np.zeros(points[idx].shape)
            im = ax.scatter(points[idx], zeros, label=legend[int(l)],
                            c=colors[i], alpha=alpha[i])
        elif(n_dim==2):
            im = ax.scatter(points[idx, 0], points[idx, 1], label=legend[int(l)],
                            c=colors[i], alpha=alpha[i])

    if(display_legend):
        ax.legend()
    if("equal_axis" in kwargs and kwargs["equal_axis"]==True):
        ax.axis('equal')
    if("savepath" in kwargs):
        plt.savefig(kwargs["savepath"])

    return fig, ax


def generate_data(n_points=500, n_classes=2, n_dim=2, cluster_std=0.3, random_state=13):
    """
    Generating a synthetic dataset by generating some blobs and transforming them so that
    they are alongated along one direction. The data is initially generated as some (gaussian) clusters.
    Then, an anisotropic transformation is applied to reshape the clusters to have a dominant direction.

    Args:
    -----
    n_points: integer
        total number of points in the dataset. All classes will have same amount of points
    n_classes: integer
        number of classes to generate
    n_dim: integer
        dimensionality of the data space
    cluster_std: float or list of floats
        standard deviation of each class cluster
    transform: string
        type of anisotropic transformation to be applied ['parallel', 'overlap']

    Returns:
    --------
    X: np array
        array with the synthetically generated data points
    y: np array
        array with the labels of the data points
    """

    X, y = make_blobs(n_samples=n_points, centers=n_classes, n_features=n_dim,
                      cluster_std=cluster_std, random_state=random_state)

    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X = np.dot(X, transformation)#

    return X, y


def main():
    """
    Main logic for generating the figure displaying POC
    """

    # generating toy dataset and saving it
    points, labels = generate_data(n_points=500, n_classes=2,
                                   cluster_std=[1, 1],
                                   random_state=22)
    display_space(points=points, labels=labels, n_dim=2, title="Synthetic Dataset",
                  savepath=os.path.join("plots", "poc_fig_original.png"),equal_axis=True)

    # computing k-means
    clusterer = KMeans(n_clusters=2, random_state=22)
    clusterer = clusterer.fit(points)
    preds = clusterer.labels_
    labels_ = np.ones(len(labels))*-1
    idx = np.where(preds==0)[0]
    labels_[idx] = 1
    idx = np.where(preds==1)[0]
    labels_[idx] = 0
    display_space(points=points, labels=labels_, n_dim=2, title=r"$k$-Means",
                  savepath=os.path.join("plots", "poc_fig_kmeans.png"),equal_axis=True)

    #
    idx1 = np.where(labels == 0)[0]
    points_1 = points[idx1, :]
    idx2 = np.where(labels == 1)[0]
    points_2 = points[idx2, :]

    # projecting dataset using POC
    poc = POC()
    poc.fit(data=points_1)
    proj_data = poc.transform(data=points, n_dims=1)
    display_space(points=proj_data, labels=labels, n_dim=1, title=r"POC",
                  savepath=os.path.join("plots", "poc_aux.png"), equal_axis=True)

    # displaying proj. data
    mean = np.mean(points, axis=0)
    eig_vects = poc.eigenvectors
    print(eig_vects[:,0])
    print(eig_vects[:,1])
    eig_vects[:,1] = [-0.5, -0.3]
    data_expand = np.zeros((proj_data.shape[0], eig_vects.shape[1]))
    data_expand[:,1:] = proj_data
    proj_data_ = np.matmul(data_expand, eig_vects)
    proj_data_ += mean
    r'$\sin (x)$'
    fig, ax = display_space(points=proj_data_, labels=labels, n_dim=2, title=r"$k$-Means +  POC",
                            equal_axis=True)
                  #savepath=os.path.join("plots", "poc_aux_2.png"))
    display_space(points=points, labels=labels, n_dim=2, title=r"POC + $k$-Means", equal_axis=True,
                  savepath=os.path.join("plots", "poc_aux_2.png"), ax=ax, fig=fig, alpha=[0.1, 0.1])


    # displaying k-means + cluster results
    clusterer = clusterer.fit(proj_data)
    preds = clusterer.labels_
    display_space(points=points, labels=preds, n_dim=2, title=r"$k$-Means + POC",
                  savepath=os.path.join("plots", "poc_fig_kmeans+poc.png"), equal_axis=True)


    return

if __name__ == "__main__":
    os.system("clear")
    main()
