
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
import numpy as np

class VotingClassifier:
    """
    Tests various machine learning models given as inputs.
    """

    def __init__(self, models:list=[]):
        """
        Stores variables

        Parameters
        ----------
        :param models: A list of machine learning algorithms to test against given datasets
        """
        self.models = models


    def fit(self, dataset:Dataset) -> 'VotingClassifier':
        """
        Trains each model with the input data

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to train each model.
        """
        for m in self.models:
            m.fit(dataset)
        return self



    def predict(self, dataset:Dataset) -> np.array:
        """
        Uses each model to predict de dependent variable, and returns an array of the most voted output variables.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        pred_vals = []
        for m in self.models:
            pred_vals.append(m.predict(dataset))

        pred_vals = np.array(pred_vals).T

        votes = []
        for pred in pred_vals:
            vals, counts = np.unique(pred, return_counts=True)
            votes.append(vals[np.argmax(counts)])

        return np.array(votes)



    def score(self, dataset:Dataset):
        """
        Returns the accuracy value for the most voted predicted variables

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        return accuracy(self.predict(dataset), dataset.y)
