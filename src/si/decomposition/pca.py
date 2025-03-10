import numpy as np
from si.data.dataset import Dataset
from sklearn import preprocessing

class PCA:
    """
    Performs the Principal Component Analysis (PCA) on a given "Dataset" object,
    using the Singular Value Decomposition (SVD) method.
    """
    def __init__(self, n_components:int):
        """
        Variable storage.

        Parameters
        ----------
        :param n_components: Number of components to be returned from the analysis
        """
        self.n_components = n_components



    def fit(self, dataset:Dataset, scale:bool = True):
        """
        Stores the mean values of each sample, the first n
        principal components (specified by the user), and their respective explained variance.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X

        # Mean values of each sample
        self.mean_vals = np.mean(data, axis=0)

        cent_data = np.subtract(data, self.mean_vals)
        U, S, Vt = np.linalg.svd(cent_data, full_matrices=False)

        # Principal components
        self.principal_comp = Vt[:self.n_components]

        # Explained variance
        n = len(dataset.X[:, 0])
        EV = (S ** 2) / (n - 1)
        self.explained_variance = EV[:self.n_components]
        return self



    def transform(self, dataset:Dataset, scale:bool = True) -> np.array:
        """
        Returns the calculated reduced SVD.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X

        cent_data = np.subtract(data, self.mean_vals)
        V = self.principal_comp.T

        # SVD reduced
        X_red = np.dot(cent_data, V)
        return X_red
