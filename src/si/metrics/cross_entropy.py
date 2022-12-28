import numpy as np

def cross_entropy(y_true, y_pred):
    """
    The cross entropy error metric for a given model.

    Parameters
    ----------
    :param y_true: The true values of the dependent variable
    :param y_true: The predicted values of the dependent variable
    """
    return -np.sum(y_true*np.log(y_pred))/len(y_true)


def cross_entropy_derivative(y_true, y_pred):
    """
    The derivative of the cross entropy error metric equation for a given model.

    Parameters
    ----------
    :param y_true: The true values of the dependent variable
    :param y_true: The predicted values of the dependent variable
    """
    return -y_true / (len(y_true)*y_pred)
