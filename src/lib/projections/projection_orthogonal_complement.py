###########################################################
# Methods for the 'Projection onto Orthogonal Complement algorithm'
# and auxiliary methods
# Scattering_Space/lib/projections
###########################################################

import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

import lib.data_loader.toy_dataset_generator as toy_dataset_generator
import lib.projections.dimensionality_reduction as dimensionality_reduction


def extract_cluster_features(data, labels, cluster_id=0, prot_method="mean"):
    """
    Computing the relevant features: prototype, eigenvectors and
    eigenvalues for a particular cluster

    Args:
    -----
    data: numpy array
        matrix containing the data points as columns (N, dims)
    labels: numpy array/list
        list with the label of each data point (N)
    cluster_id: integer
        number of the cluster we want to compute the statistics from
    prot_method: string
        statistic used to compute the class prototype ['mean', 'median']

    Returns:
    --------
    prototype: numpy array
        class prototype. Corresponds to the mean/median of the class
    eigenvectors: numpy array
        eigenvector matrix of the class data-matrix, eigenvectors are sorted
        in descending order of spanned variance
    eigenvalues: numpy array
        eigenvalues from the data matrix sorted in descending order
    """

    assert prot_method in ["mean", "median"]

    # reshaping input data to (N, dims)
    if(len(data.shape) > 2):
        data = data.reshape(data.shape[0], -1)

    if(cluster_id not in labels):
        print(f"There are no data points with label: {cluster_id}")
        return None, None

    # sampling relevant data
    idx = np.where(labels == cluster_id)[0]
    classwise_data = data[idx, :]

    # computing class prototype
    if(prot_method == "mean"):
        prototype = np.mean(classwise_data, axis=0)
    elif(prot_method == "median"):
        prototype = np.median(classwise_data, axis=0)

    # eigenvalue decomposition
    standardized_data = StandardScaler().fit_transform(classwise_data)
    data_matrix = np.cov(standardized_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(data_matrix)

    # sorting by magnitude
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:,idx].astype(float)
    eigenvalues = eigenvalues[idx].astype(float)

    return prototype, eigenvectors, eigenvalues


def compute_prototype(data, stat="mean"):
    """
    Computing the mean/mediod of the point cloud, which corresponds to the
    class prototype

    Args:
    -----
    data: numpy array
        array-like object with the dataset in scattering or pixel domain
    stat: string
        statistic used to determine the prototype. Can be 'mean' or 'median'

    Returns:
    --------
    prototype: np array
        data point corresponding to the class prototype
    """

    if(stat=="mean"):
        prototype = np.mean(data, axis=0)
    elif(stat=="median"):
        prototype = np.median(data, axis=0)
    else:
        print(f"Statistic {stat} is not valid. Use one of the following ['mean', 'median']")
        assert False
    prototype = prototype[np.newaxis,:]

    return prototype


def projection_onto_orthogonal_complement(data, eigenvectors, n_directions):
    """
    Projecting datapoints onto the orthogonal complement of the main directions of variance

    Args:
    -----
    data: numpy array
        array containing the data points to project
    eigenvectors: numpy array
        matrix containing the eigenvectors as columns. These are sorted, thus being column 1 the
        eigenvector corresponding to the direction of largest data variance
    n_directions: integer
        number of eigenvectors to remove prior to projecting the data

    Returns:
    --------
    projected_data: np array
        projections of the input data onto the lower-dim orthogonal complement
    """

    eigenvectors_reduced = eigenvectors[:, n_directions:]  # removing the eigenvectors associated to the largest eigenvalues
    projected_data = np.matmul(eigenvectors_reduced.T, data)  # projecting onto low-dim space

    return projected_data


def projection_onto_eigenvectors(data, eigenvectors, n_directions, orthogonal=True):
    """
    Projecting datapoints onto the main directions of variance to recovere the original basis

    Args:
    -----
    data: numpy array
        array containing the data points to project
    eigenvectors: numpy array
        matrix containing the eigenvectors as columns. These are sorted, thus being column 1 the
        eigenvector corresponding to the direction of largest data variance
    n_directions: integer
        number of eigenvectors to remove prior to project
    orthogonal: Boolean
        If True, orthogonal complement experiment was run and dominarn dimensions should added
        If False, last directions should be extended

    Returns:
    --------
    projected_data: np array
        projections of the input data onto the lower-dim manifold spanned by largest eigenvectors
        This is pretty much the same as PCA
    """

    if(len(data.shape)==1):
        data = data[np.newaxis, :]
    data_expand = np.zeros((eigenvectors.shape[1], data.shape[1]))
    if(orthogonal):
        data_expand[n_directions:,:] = data
    else:
        data_expand[:n_directions,:] = data
    projected_data = np.matmul(eigenvectors, data_expand)

    return projected_data


def compute_projections_dominant(points, eigenvectors, prototypes, n_directions=1, debug=False):
    """
    Projecting points into the space spanned by the dominant eigenvectors

    Args:
    -----
    points: np array
        array with the points to project
    eigenvectors: list of np arrays
        list with the eigenvector matrix for each class that we want to project onto
    prototypes: list of np arrays
        list with the class-prototype for each class that we want to project onto
    n_directions: int
        number of directions to remove.

    Returns:
    --------
    all_projections: list
        list containing the projections onto the dominatn directions
    """

    all_projections = []

    if(len(points.shape)>2):
        points = points.reshape(points.shape[0], -1)

    for i in range(len(prototypes)):

        # shifting pointcloud to origin
        points_mean_free = points - prototypes[i]
        prototype_mean_free = prototypes[i]  - prototypes[i]

        # projecting onto dominant eigenvectors
        eigenvectors_reduced = eigenvectors[i][:, :n_directions]
        proj_points_mean_free = np.matmul(eigenvectors_reduced.T, points_mean_free.T).T
        proj_prototype_mean_free = np.matmul(eigenvectors_reduced.T, prototype_mean_free.T).T

        # recovering original basis
        proj_points_recovered = projection_onto_eigenvectors(proj_points_mean_free.T, eigenvectors=eigenvectors[i],
                                                             n_directions=n_directions, orthogonal=False).T
        proj_prototype_recovered = projection_onto_eigenvectors(proj_prototype_mean_free.T, eigenvectors=eigenvectors[i],
                                                                n_directions=n_directions, orthogonal=False).T
        # recovering offset
        points_recovered = proj_points_recovered + prototypes[i]
        prototype_recovered = proj_prototype_recovered + prototypes[i]
        all_projections.append(points_recovered)

    return all_projections


def compute_projections(points, eigenvectors, prototypes, n_directions=1, debug=False):
    """
    Projecting points into (possibly many) othogonal complement space using the eigenvectors for projection
    and the prototype for shifting the space to the origin of coordinates

    Args:
    -----
    points: np array
        array with the points to project
    eigenvectors: list of np arrays
        list with the eigenvector matrix for each class that we want to project onto
    prototypes: list of np arrays
        list with the class-prototype for each class that we want to project onto
    n_directions: int
        number of directions to remove.

    Returns:
    --------
    all_projections: list
        list containing the projections onto the orthogonal complement
    """

    all_projections = []

    if(len(points.shape)>2):
        points = points.reshape(points.shape[0], -1)

    for i in range(len(prototypes)):

        # shifting pointcloud to origin
        points_mean_free = points - prototypes[i]
        prototype_mean_free = prototypes[i]  - prototypes[i]

        # changing basis
        if(debug):
            proj_points_mean_free_test = projection_onto_orthogonal_complement(points_mean_free.T, eigenvectors=eigenvectors[i],
                                                                               n_directions=0).T
            proj_prototype_mean_free_test = projection_onto_orthogonal_complement(prototype_mean_free.T,
                                                                                  eigenvectors=eigenvectors[i],
                                                                                  n_directions=0).T

        # projecting onto orthogonal complement
        proj_points_mean_free = projection_onto_orthogonal_complement(points_mean_free.T, eigenvectors=eigenvectors[i],
                                                                      n_directions=n_directions).T
        proj_prototype_mean_free = projection_onto_orthogonal_complement(prototype_mean_free.T, eigenvectors=eigenvectors[i],
                                                                         n_directions=n_directions).T
        # recovering original basis
        proj_points_recovered = projection_onto_eigenvectors(proj_points_mean_free.T, eigenvectors=eigenvectors[i],
                                                             n_directions=n_directions).T
        proj_prototype_recovered = projection_onto_eigenvectors(proj_prototype_mean_free.T, eigenvectors=eigenvectors[i],
                                                                n_directions=n_directions).T
        # recovering offset
        points_recovered = proj_points_recovered + prototypes[i]
        prototype_recovered = proj_prototype_recovered + prototypes[i]
        all_projections.append(points_recovered)

        if(debug):
            return points, points_mean_free, proj_points_mean_free_test, proj_points_mean_free,\
                   proj_points_recovered, points_recovered

    return all_projections


def projections_classifier(projections, prototypes, metric="euclidean"):
    """
    Computing classification based on a distance metric between projections and prototypes

    Args:
    -----
    projections: list of np array
        list containing the projections of the data onto each of the classes
    prototypes: list of np array
        list containing the prototypes of each class
    metric: string
        type of distance to be used {manhattan, euclidean, minkowski, ...}

    Returns:
    --------
    labels: np array
        array containing the predicted labels for each of the test points
    min_distances: np array
        distance between each test point and the corresponding closest prototype
    """

    if(len(projections)!=len(prototypes)):
        print("Prjections list and Prototypes list must have same length!")
        return None

    # computing all distances to prototypes
    distances = []
    for i in range(len(prototypes)):
        if(len(prototypes[i].shape)==1):
            prototypes[i] = prototypes[i][np.newaxis,:]
        current_distances = metrics.pairwise_distances(projections[i], prototypes[i], metric=metric)
        distances.append(current_distances)

    # chosing the labels corresponding to the smallest distances
    distances = np.array(distances)[:,:,0]
    labels = np.argmin(distances, axis=0)
    min_distances = np.min(distances, axis=0)

    return labels, min_distances


def projections_classifier_(points, eigenvectors, prototypes, n_directions):
    """
    Best method to apply the algortihm for classification
    """

    distances = []

    if(len(points.shape)>2):
        points = points.reshape(points.shape[0], -1)

    for i in range(len(prototypes)):
        eigenvectors_reduced = eigenvectors[i][:, n_directions:]  # removing the eigenvectors associated to the largest eigenvalues
        # current_distances = metrics.pairwise_distances(points, prototypes[i], metric="euclidean")
        current_distances = points - prototypes[i]
        projected_distances = np.matmul(eigenvectors_reduced.T, current_distances.T).T
        projected_distances = np.linalg.norm(projected_distances, axis=1)
        distances.append(projected_distances)

    # chosing the labels corresponding to the smallest distances
    distances = np.array(distances)
    labels = np.argmin(distances, axis=0)
    min_distances = np.min(distances, axis=0)

    return labels, min_distances


def display_projection_steps(train_points, prototype, test_points, eigenvectors, n_directions=1,
                             c="red", equal_axis=True):
    """
    Displaying all intermediate projection steps for visualization and explanatory purposes

    Args:
    -----
    train_points: numpy array
        points belonging to the training set
    prototype: numpy array
        class prototypes
    test_points: numpy array
        points belonging to the test set
    eigenvectors: numpy array
        eigenvectors corresponding to the different training set classes
    n_directions: int
        number of directions to remove
    c: string
        color to use for the training set points
    equal_axis: boolean
        If true, aspect ratio is maintained
    """

    # obtaining train point projections
    train_points, train_points_mean_free, train_proj_points_mean_free_test,\
        train_proj_points_mean_free, train_proj_points_recovered,\
        train_points_recovered = compute_projections(train_points, [eigenvectors], [prototype], n_directions, debug=True)

    # obtaining test point projections
    test_points, test_points_mean_free, test_proj_points_mean_free_test,\
        test_proj_points_mean_free, test_proj_points_recovered,\
        test_points_recovered = compute_projections(test_points, [eigenvectors], [prototype], n_directions, debug=True)

    # obtaining prototype projections
    prot_points, prot_points_mean_free, prot_proj_points_mean_free_test,\
        prot_proj_points_mean_free, prot_proj_points_recovered,\
        prot_points_recovered = compute_projections(prototype, [eigenvectors], [prototype], n_directions, debug=True)

    # eigenvectors for plotting
    xaxis = np.arange(-3,3,0.05)
    xaxis_shift = xaxis + prototype[:,0]
    eig1 = xaxis*eigenvectors[1,0]/eigenvectors[0,0]
    eig2 = xaxis*eigenvectors[1,1]/eigenvectors[0,1]
    eig1_shift = eig1 + prototype[:,1]
    eig2_shift = eig2 + prototype[:,1]

    plt.figure(figsize=(18,10))
    plt.subplot(231)
    plt.plot(xaxis_shift, eig1_shift, label="Eigenvector", c="black")
    plt.plot(xaxis_shift, eig2_shift, c="black")
    plt.plot([prototype[:,0], prototype[:,0]+eigenvectors[0,0]], [prototype[:,1], prototype[:,1]+eigenvectors[1,0]],
         linewidth=5, c="black")
    plt.plot([prototype[:,0], prototype[:,0]+eigenvectors[0,1]], [prototype[:,1], prototype[:,1]+eigenvectors[1,1]],
             linewidth=5, c="black")
    plt.scatter(train_points[:, 0], train_points[:, 1], label="Train Points", c=c)
    plt.scatter(prot_points[:, 0], prot_points[:, 1], label="Class Prototype", c="green")
    plt.scatter(test_points[:, 0], test_points[:, 1], label="Test Points", c="orange")
    plt.title("Original Data")
    if(equal_axis):
        plt.axis('equal')
    plt.legend()

    plt.subplot(2,3,2)
    plt.scatter(train_points_mean_free[:, 0], train_points_mean_free[:, 1], c=c)
    plt.scatter(prot_points_mean_free[:, 0], prot_points_mean_free[:, 1], c="green")
    plt.scatter(test_points_mean_free[:, 0], test_points_mean_free[:, 1], c="orange")
    plt.title("Shifted Data")
    if(equal_axis):
        plt.axis('equal')

    plt.subplot(2,3,3)
    plt.scatter(train_proj_points_mean_free_test[:, 0], train_proj_points_mean_free_test[:, 1], c=c)
    plt.scatter(prot_proj_points_mean_free_test[:, 0], prot_proj_points_mean_free_test[:, 1], c="green")
    plt.scatter(test_proj_points_mean_free_test[:, 0], test_proj_points_mean_free_test[:, 1], c="orange")
    proj_eigenvectors_0 = projection_onto_orthogonal_complement(eigenvectors[:,0], eigenvectors=eigenvectors,
                                                                n_directions=0).T
    proj_eigenvectors_1 = projection_onto_orthogonal_complement(eigenvectors[:,1], eigenvectors=eigenvectors,
                                                                n_directions=0).T
    plt.title("Change of Basis: product with full eigenvector matrix")
    if(equal_axis):
        plt.axis('equal')

    plt.subplot(2,3,4)
    plt.scatter(train_proj_points_mean_free_test[:, 0], train_proj_points_mean_free_test[:, 1], c=c, alpha=0.2)
    plt.scatter(prot_proj_points_mean_free_test[:, 0], prot_proj_points_mean_free_test[:, 1], c="green", alpha=0.2)
    plt.scatter(test_proj_points_mean_free_test[:, 0], test_proj_points_mean_free_test[:, 1], c="orange", alpha=0.2)
    plt.scatter(np.zeros(train_proj_points_mean_free.shape), train_proj_points_mean_free[:, 0], c=c)
    plt.scatter(np.zeros(test_proj_points_mean_free.shape), test_proj_points_mean_free[:, 0], c="orange")
    plt.scatter(np.zeros(prot_proj_points_mean_free.shape), prot_proj_points_mean_free[:, 0], c="green")
    plt.title("Projection onto orthogonal complement")
    if(equal_axis):
        plt.axis('equal')

    plt.subplot(2,3,5)
    plt.scatter(train_points_mean_free[:, 0], train_points_mean_free[:, 1], c=c, alpha=0.2)
    plt.scatter(prot_points_mean_free[:, 0], prot_points_mean_free[:, 1], c="green", alpha=0.2)
    plt.scatter(test_points_mean_free[:, 0], test_points_mean_free[:, 1], c="orange", alpha=0.2)
    plt.scatter(train_proj_points_recovered[:, 0], train_proj_points_recovered[:, 1], c=c)
    plt.scatter(prot_proj_points_recovered[:, 0], prot_proj_points_recovered[:, 1], c="green")
    plt.scatter(test_proj_points_recovered[:, 0], test_proj_points_recovered[:, 1], c="orange")
    plt.title("Recovering original Basis")
    if(equal_axis):
        plt.axis('equal')

    plt.subplot(2,3,6)
    plt.scatter(train_points[:, 0], train_points[:, 1], c=c, alpha=0.2)
    plt.scatter(prot_points[:, 0], prot_points[:, 1], c="green", alpha=0.2)
    plt.scatter(test_points[:, 0], test_points[:, 1], c="orange", alpha=0.2)
    plt.plot(xaxis_shift, eig1_shift, c="black", alpha=0.5)
    plt.plot(xaxis_shift, eig2_shift, c="black", alpha=0.5)
    plt.scatter(train_points_recovered[:, 0], train_points_recovered[:, 1], label="Train Data", c=c)
    plt.scatter(prot_points_recovered[:, 0], prot_points_recovered[:, 1], label="Class Prototype", c="green")
    plt.scatter(test_points_recovered[:, 0], test_points_recovered[:, 1], label="Test Point", c="orange")
    plt.title("Recovering offset")
    if(equal_axis):
        plt.axis('equal')

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    return


def classification_experiment(train_data, test_data, train_labels, test_labels, n_directions=1,
                              eigenvectors=None, prototypes=None, verbose=1):
    """
    Computing a complete classification experiment

    Args:
    -----
    train_data: np array
        data used for computing the prototype
    test_data: np array
        test data used to assess the method
    train_labels: np array
        labels corresponding to the training set
    test_labels: np array
        labels corresponding to the test set.
    n_directions: integer
        number of directions to remove when computing the orthogonal complement
    eigenvectors: list
        list containing the eigenvectors (sorted by relevance) corresponding to each class
        If not given, they will be computed
    prototypes: list
        list containing the prototype array for each class
        If not given, they will be computed
    verbose: integer
        verbosity level

    Returns:
    --------
    accuracy: float
        classification accuracy on the test set
    distances: list
        distances between projected point and closest prototype
    classwise_vars: dictionary
        dictionary containing classwise prototypes and eigenvectors. It is used
        to avoid recomputing these vectors.
    """

    n_labels = len(np.unique(train_labels))

    # computing class prototypes and eigendecomposition if necessary
    if(eigenvectors is None or prototypes is None):
        eigenvalues = []
        eigenvectors = []
        prototypes = []
        for i in range(n_labels):
            class_data = toy_dataset_generator.get_classwise_data(data=train_data, labels=train_labels, label=i, verbose=0)
            class_prototype = compute_prototype(class_data).flatten()
            prototypes.append(class_prototype)
            _, class_eigenvectors = dimensionality_reduction.compute_eigendecomposition(class_data)
            eigenvectors.append(class_eigenvectors)
    classwise_vars = {"eigenvectors":eigenvectors, "prototypes":prototypes}

    # projecting prototypes and test data
    # test_set_proj = compute_projections(test_data, eigenvectors=eigenvectors, prototypes=prototypes, n_directions=n_directions)
    # predicting labels for the test set
    # predicted_labels, distances = projections_classifier(test_set_proj, prototypes)
    predicted_labels, distances = projections_classifier_(test_data, eigenvectors=eigenvectors, prototypes=prototypes,
                                                          n_directions=n_directions)


    # computing accuracy
    correct_classifications = len(np.where(test_labels==predicted_labels)[0])
    num_points = len(test_labels)
    accuracy = 100*correct_classifications/num_points
    if(verbose>0):
        print(f"Results for {n_directions} removed directions:")
        print(f"    Classification accuracy on test set is {accuracy}%")

    # additional distance information
    if(verbose>1):
        print(f"    Average distance between projections and prototype: {np.mean(distances)}")
        idx = np.where(test_labels==predicted_labels)[0]
        print(f"    Average distance for correctly classified points: {np.mean(distances[idx])}")
        idx = np.where(test_labels!=predicted_labels)[0]
        print(f"    Average distance for wrongly classified points: {np.mean(distances[idx])}\n\n")

    return accuracy, distances, classwise_vars


#
