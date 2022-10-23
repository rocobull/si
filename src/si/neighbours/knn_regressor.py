
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse
from si.data.dataset import Dataset
import numpy as np

class KNNRegressor:
    """
    Performs the K-Nearest Neighbours algorithm for regression problems.
    """
    def __init__(self, k: int, distance=euclidean_distance):
        """
        Stores variables

        Parameters
        ----------
        :param k: Number of nearest neighbours to consider for value assignment (mean value of k nearest neighbours).
        :param distance: A function to determine distance between observations and the clusters.
                         The "euclidean_distance" function is used by default.
        """
        self.k = k
        self.distance = distance



    def fit(self, dataset: Dataset):
        """
        Stores the training dataset

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        self.dataset = dataset  # Training dataset
        return self



    def _get_closest_sample(self, sample: np.array):
        """
        Auxiliary function to assign the given sample the mean value of its nearest neighbours.

        Parameters
        ----------
        :param sample: A sample of a given dataset.
        """
        dists = self.distance(sample.astype(float), self.dataset.X.astype(float))
        closest_inds = np.argsort(dists)[:self.k]
        closest_y = self.dataset.y[closest_inds]
        return np.mean(closest_y)



    def predict(self, dataset: Dataset):
        """
        Assigns each sample in the given test dataset the mean value of their nearest neighbours.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        return np.apply_along_axis(self._get_closest_sample, axis=1, arr=dataset.X)



    def score(self, dataset: Dataset):
        """
        Gives an accuracy value between true dependent variable values and predicted values.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        return rmse(dataset.y, self.predict(dataset))

