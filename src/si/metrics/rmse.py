import numpy as np

def rmse(y_true:np.array, y_pred:np.array):
    """
    Root Mean Squared Error (RMSE) metric to use between true and predicted values.

    Parameters
    ----------
    :param y_true: Array of true values
    :param y_pred: Array of predicted values
    """
    N = len(y_true)
    return np.sqrt(np.sum(np.subtract(y_true, y_pred)**2 / N))