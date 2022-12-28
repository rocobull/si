import numpy as np
import warnings

def sigmoid_function(X:np.array) -> np.array:
    """
    Implements the sigmoid function to an array of values

    Parameters
    ----------
    :param X: Array of values
    """
    # suppress warnings
    warnings.filterwarnings('ignore')

    X = X.astype(float)
    return 1/(1+(np.exp(-X)))
