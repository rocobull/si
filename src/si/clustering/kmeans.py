import numpy as np
from si.statistics.euclidean_distance import euclidean_distance
from si.data.dataset import Dataset
from sklearn import preprocessing


class KMeans:
    """
    Class for clustering data using the KMeans method
    """

    def __init__(self, k: int, max_iter: int = 300, distance: euclidean_distance = euclidean_distance):
        """
        Stores variables

        Parameters
        ----------
        :param k: Number of clusters to create
        :param max_iter: Maximum number of iterations ti perform while calculating the centroids of each cluster
        :param distance: A function to determine distance between observations and the clusters.
                         The "euclidean_distance" function is used by default
        """
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

    def _closest_centroid(self, row: np.array, centroids: np.array):
        """
        Auxiliary method that returns the closest centroid to the chosen data row

        Parameters
        ----------
        :param row: A sample array from a dataset
        :param centroids: A list of arrays containing the coordinates for each cluster centroid
        """
        dists = self.distance(row, centroids)
        best_ind = np.argmin(dists, axis=0)
        return best_ind

    def fit(self, dataset:Dataset, scale:bool = True):
        """
        Determines best fitted data centroids according to the given dataset and number of clusters

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X
        seeds = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[seeds, :]

        convergence = True
        labels = []
        i = 0
        while convergence and i < self.max_iter:
            new_labels = np.apply_along_axis(self._closest_centroid, axis=1, arr=data,
                                             centroids=self.centroids)  # Rows (along a column)
            all_centroids = []
            for ix in range(self.k):
                centroid = data[new_labels == ix]
                cent_mean = np.mean(centroid, axis=0)  # Columns
                all_centroids.append(cent_mean)

            self.centroids = np.array(all_centroids)

            convergence = np.any(new_labels != labels)
            labels = new_labels
            i += 1

        print("Number of iterations:", i)
        return self

    def transform(self, dataset:Dataset, scale:bool = True) -> np.array:
        """
        Determines the distances of each observation to each centroid determined using the fit() method.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X

        return np.apply_along_axis(self.distance, axis=1, arr=data, y=self.centroids)

    def predict(self, dataset:Dataset, scale:bool = True) -> np.array:
        """
        Calls the transform() method and associated a cluster to each observation based on their distances
        (the observation is attributed to the closest cluster).

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        dists = self.transform(dataset, scale)
        labels = np.argmin(dists, axis=1)
        return labels
