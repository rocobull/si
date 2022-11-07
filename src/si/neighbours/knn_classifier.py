
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
import numpy as np

class KNNClassifier:
    """
    Performs the K-Nearest Neighbours algorithm for classification problems.
    """
    def __init__(self, k:int, distance=euclidean_distance):
        """
        Stores variables.

        Parameters
        ----------
        :param k: Number of nearest neighbours to consider for group assignment.
        :param distance: A function to determine distance between observations and the clusters.
                         The "euclidean_distance" function is used by default.
        """
        self.k = k
        self.distance = distance



    def fit(self, dataset:Dataset):
        """
        Stores the training dataset.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to train the model.
        """
        self.dataset = dataset #Training dataset
        return self



    def _get_closest_sample(self, sample:np.array):
        """
        Auxiliary function to assign the given sample a cluster based on its nearest neighbours.

        Parameters
        ----------
        :param sample: A sample of a given dataset.
        """
        dists = self.distance(sample.astype(float), self.dataset.X.astype(float))
        closest_inds = np.argsort(dists)[:self.k]
        closest_y = self.dataset.y[closest_inds]
        #print(np.unique(closest_y, return_counts=True))
        unique_vals, count = np.unique(closest_y, return_counts=True)
        return unique_vals[np.argmax(count)]



    def predict(self, dataset:Dataset):
        """
        Assigns a group to each sample in the given test dataset based on their nearest neighbours.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        return np.apply_along_axis(self._get_closest_sample, axis=1, arr=dataset.X)



    def score(self, dataset:Dataset):
        """
        Gives an accuracy value between true dependent variable values and predicted values.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        return accuracy(dataset.y, self.predict(dataset))

