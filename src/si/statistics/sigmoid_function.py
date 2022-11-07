import numpy as np


def sigmoid_function(X: np.array) -> np.array:
    """
    Implements the sigmoid function to an array of values

    Parameters
    ----------
    :param X: Array of values
    """
    return 1/(1+(np.exp(-X)))
