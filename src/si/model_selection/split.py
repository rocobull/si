from si.data.dataset import Dataset
import numpy as np


def train_test_split(dataset: Dataset, test_size: float, random_state: int) -> tuple:
    """
    Splits data from the given Dataset class instance into training and testing data.
    Returns 2 Dataset class instances, the corresponding to training data and the second to testing data.

    Parameters
    ----------
    :param dataset: An instance of the Dataset class.
    :param test_size: Percentage of the data to be used as test data (value from 0 to 1).
    :param random_state: Seed value for results to be reproducible.
    """
    np.random.seed(random_state)

    n_samples = dataset.shape()[0][0]

    size_test = int(n_samples * test_size)

    all_permutation = np.random.permutation(n_samples)

    test_inds = all_permutation[:size_test]
    train_inds = all_permutation[size_test:]

    train_X = dataset.X[train_inds]
    train_y = dataset.y[train_inds]
    test_X = dataset.X[test_inds]
    test_y = dataset.y[test_inds]

    train_data = Dataset(train_X, train_y, dataset.features, dataset.label)
    test_data = Dataset(test_X, test_y, dataset.features, dataset.label)

    return train_data, test_data
