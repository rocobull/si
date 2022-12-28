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




if __name__ == "__main__":
    true = np.array([0, 1, 1, 1, 0, 1])
    pred = np.array([1, 0, 1, 1, 0, 1])
    print(f"RMSE: {rmse(true, pred):.4f}")