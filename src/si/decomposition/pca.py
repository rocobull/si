import numpy as np


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



    def fit(self, dataset:object) -> tuple:
        """
        Stores the mean values of each sample, the first n
        principal components (specified by the user), and their respective explained variance.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        # Mean values of each sample
        self.mean_vals = np.mean(dataset.X, axis=0)

        self.cent_data = np.subtract(dataset.X, self.mean_vals)
        U, S, Vt = np.linalg.svd(self.cent_data, full_matrices=False)

        #self.X = U * S * Vt

        # Principal components
        self.principal_comp = Vt[:self.n_components]

        # Explained variance
        n = len(dataset.X[:, 0])
        EV = (S ** 2) / (n - 1)
        self.explained_variance = EV[:self.n_components]
        return self



    def transform(self, dataset: object) -> tuple:
        """
        Returns the calculated reduced SVD.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        V = self.principal_comp.T
        # SVD reduced
        X_red = np.dot(self.cent_data, V)

        return X_red
