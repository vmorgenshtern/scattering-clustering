"""
Custom imlementation of the ultra-scalable spectral clustering algorithm (USPEC)
 - 'Ultra-Scalable Spectral Clustering and Ensemble Clustering' Huang et al. 2019
"""

from time import time
import numpy as np
import scipy as sp
import hnswlib
from sklearn.cluster import KMeans, MiniBatchKMeans

from lib.dimensionality_reduction.dimensionality_reduction import compute_eigendecomposition
from lib.utils.logger import log_function, print_
from lib.utils.utils import for_all_methods

@for_all_methods(log_function)
class USPEC():
    """
    Custom imlementation of the ultra-scalable spectral clustering algorithm (USPEC).
    This method is able to scale standard spectral clustering to datasets composed
    by millions of samples without losing significant performance. O(N log N)
    """

    def __init__(self, p_interm=5e4, p_final=5e2, n_neighbors=5,
                 num_clusters=10, num_iters=100, random_seed=0):
        """
        Initializer of the clustering method

        Args:
        -----
        p_interm: integer
            Number of random samples taken as candidates for the representatives
            Sampled at random from the dataset
        p_final: integer
            Total number of representatives sampled from the dataset.
            Obtained using k_means on the p_interm data points
        n_neighbors: integer
            Number of n_neighbors to consider to compute the affinity submatrix
        num_clusters: integer
            number of clusters to extract from the data
        num_iters: integer
            maximum number of iterations to compute in the K-means algorithm
        """

        self.random_seed = random_seed
        np.random.seed(seed=random_seed)

        # parameters for represnetative sampling
        self.p_interm = p_interm
        self.p_final = p_final

        # parameters for computing affinity submatrix
        self.n_neighbors = n_neighbors

        # parameters for the transfer cut algorithm
        self.num_clusters = num_clusters
        self.num_iters = num_iters

        return


    def cluster(self, data, verbose=0):
        """
        Performing the complete clustering flow for the given data samples

        Args:
        -----
        data: numpy array
            Array containing the dataset. Shape is (N, d)
        verbose: integer
            verbosity level

        Returns:
        --------
        labels: numpy array
            array with the predicted cluster labels for each data sample. Remember that
            cluster labels need not match the original labels.
        """

        if(verbose > 0):
            print_(f"Starting clustering:")
            step0 = time()

        reps, cands = self.sample_representatives(data)

        if(verbose > 0):
            step1 = time()
            print_(f"  Sampling time: {round(step1-step0, 2)} seconds")

        graph = self.compute_affinity_submatrix(reps=reps, data=data)

        if(verbose > 0):
            step2 = time()
            print_(f"  Computing affinity time: {round(step2-step1, 2)} seconds")

        labels = self.transer_cut_bipartite(graph=graph)

        if(verbose > 0):
            step3 = time()
            print_(f"  Transfer cut time: {round(step3-step2, 2)} seconds")

        return labels


    def sample_representatives(self, data, p_interm=None, p_final=None):
        """
        Sampling the representative samples for the affininity submatrix

        Args:
        -----
        data: numpy array
            Array containing the dataset. Shape is (N, d)
        p_interm: integer
            Number of random samples to take from the dataset
            Sampled at random from the dataset
        p_final: integer
            Total number of representatives sampled from the dataset.
            Obtained using k_means on the p_interm data points

        Returns:
        -------
        representatives: numpy array
            Matrix with the representative samples. Shape is (p_final, n_dim)
        candidates: numpy array
            Matrix with the candidate samples. Shape is (p_interm, n_dim)
        """

        # processign parameters, checking for changed values
        if(p_interm is not None):
            self.p_interm = p_interm
        if(p_final is not None):
            self.p_final = p_final

        # enforcing correct values for the parameters
        assert data.shape[0] > self.p_interm, f"'p_interm: {self.p_interm}' must be " \
            f"smaller than thenumber of samples in the dataset: {data.shape[0]}"
        assert self.p_interm > self.p_final, f"'p_final: {self.p_final}' must be " \
            f"smaller than the  number of intermediate samples 'p_interm': {self.p_interm}"

        # step 1: randomly sampling the representative candidates
        N = data.shape[0]
        rand_idx = np.random.randint(low=0, high=N, size=int(self.p_interm))
        candidates = data[rand_idx, :]

        # step 2: obtainig representatives using k-meanss
        # kmeans = KMeans(n_clusters=self.num_clusters, max_iter=100,
        #                 n_jobs=-1, random_state=self.random_seed).fit(candidates)
        kmeans = MiniBatchKMeans(n_clusters=int(self.p_final), max_iter=100,
                                 batch_size=int(self.p_interm//20),
                                 random_state=self.random_seed).fit(candidates)
        representatives = kmeans.cluster_centers_

        return representatives, candidates


    def compute_affinity_submatrix(self, reps, data, n_neighbors=None):
        """
        Computing an affinity submatrix between the representative samples and the
        complete dataset

        Args:
        -----
        reps: numpy array
            Matrix with the representative samples. Shape is (p, n_dim)
        data: numpy array
            Matrix with the complete training set. Shape is (N, n_dim)
        n_neighbors: integer
            Number of n_neighbors to consider to compute the affinity submatrix

        Returns:
        --------
        affinity_matrix: numpy array
            Matrix with shape (n_samples, p_final) which informs about the connectivity
            of the dataset samples with the representatives => bipartite graph
        """

        num_samples = data.shape[0]
        num_reps, data_dim = reps.shape
        if(n_neighbors != None):
            self.n_neighbors = n_neighbors

        """
        In original paper they use a different approach for the kNN. Nevertheless, we
        rely on the HNSW graphs, which shows high efficiency for high-dim feature vectors
        """
        # creating hns graph object and fitting features
        m = 8              # minimum number of outgoing edges for each node
        ef = 1000           # increase query "volume" to make results more stable
        neighbors_graph = hnswlib.Index(space='l2', dim=data_dim)
        neighbors_graph.init_index(max_elements=num_reps, ef_construction=ef, M=m)
        neighbors_graph.set_ef(ef)
        neighbors_graph.add_items(reps, np.arange(num_reps))

        # retrieving kNN for each data-point and fitting affinity matrix
        affinity_matrix = 1e10 * np.ones((num_samples, num_reps))
        for i in range(num_samples):
            idx, distance = neighbors_graph.knn_query(data[i,:], k=self.n_neighbors)
            affinity_matrix[i, idx] = distance

        # converting distance into affinities using a heat kernel
        sigma = np.mean(affinity_matrix[affinity_matrix < 1e9])
        affinity_matrix =  np.exp( - (affinity_matrix*affinity_matrix) / (2*sigma*sigma))

        return affinity_matrix


    def transer_cut_bipartite(self, graph, num_clusters=None, num_iters=None):
        """
        Predicting cluster assignments for each sample using the transfer cut algorithm
         - 'Segmentation Using Superpixels: A Bipartite Graph Partitioning Approach'
           Li et al., CVPR 2012

        Args:
        -----
        graph: numpy array
            bipartite graph given as an affinity (sub)matrix with shape (n_samples, p_final)
        num_clusters: integer
            number of clusters to obtain
        num_iters: integer
            number of iterations for the K-Means algorithm

        Returns:
        --------
        labels: numpy array
            array with the predicted cluster labels for each data sample. Remember that
            cluster labels need not match the original labels.
        """

        eps = 1e-10
        if(num_clusters != None):
            self.num_clusters = num_clusters
        if(num_iters != None):
            self.num_iters = num_iters

        # transfer-cut transform of the affinity matrix
        idx = np.arange(graph.shape[0])
        dx = 1/(np.sum(graph, axis=1) + eps)
        Dx = sp.sparse.csr_matrix((dx, (idx, idx)))
        Wy = graph.T @ Dx @ graph

        # converting affinity into graph laplacian
        idx = np.arange(Wy.shape[0])
        d = 1/np.sqrt(np.sum(Wy, axis=1) + eps)
        D = sp.sparse.csr_matrix((d, (idx, idx)))
        nWy = D @ Wy @ D
        nWy = (nWy.T + nWy) / 2

        # eigendecomposition
        eigenvalues, eigenvectors = compute_eigendecomposition(nWy, standarize=False)
        norm_cut_eigenvec = D @ eigenvectors[:,:self.num_clusters]

        # compiting eigenvectors on the entire bipartite graph
        all_eigenvec = Dx @ graph @ norm_cut_eigenvec
        all_eigenvec = all_eigenvec / (np.linalg.norm(all_eigenvec,
                                       axis=1, keepdims=True) + eps)

        # k-means
        kmeans = KMeans(n_clusters=self.num_clusters, max_iter=self.num_iters,
                        n_jobs=-1, random_state=self.random_seed)
        kmeans.fit(all_eigenvec)
        labels = kmeans.labels_

        return labels


    def compute_eigengap(self, graph, alpha=100):
        """
        Computing the eigendecomposition of the graph laplacian to measure the eigenga
        and to determine an estimate of the number of clusters
        """

        eigenvals, _ = compute_eigendecomposition(graph, standarize=False)
        # eigenvals = eigenvals[::]

        # computing largest eigengap
        if(alpha > 0):
            eigenvals = eigenvals[:alpha]
        diff_eigenvals = np.diff(eigenvals)[1:]
        eigengap = np.argmax(diff_eigenvals) + 2

        return eigenvals, eigengap

#
