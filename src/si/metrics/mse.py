import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    The mean squared error metric for a given model.

    Parameters
    ----------
    :param y_true: The true values of the dependent variable
    :param y_true: The predicted values of the dependent variable
    """
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)


def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray):
    """
    The derivative of the mean squared error metric equation for a given model.

    Parameters
    ----------
    :param y_true: The true values of the dependent variable
    :param y_true: The predicted values of the dependent variable
    """
    return -(y_true - y_pred) / len(y_true)
