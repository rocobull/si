import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse
from sklearn import preprocessing

class RidgeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    :param l2_penalty: The L2 regularization parameter
    :param alpha: The learning rate
    :param max_iter: The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        """
        Stores variables.

        Parameters
        ----------
        :param l2_penalty: The L2 regularization parameter
        :param alpha: The learning rate
        :param max_iter: The maximum number of iterations
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}



    def fit(self, dataset: Dataset, use_adaptive_fit: bool = False, scale:bool = True) -> 'RidgeRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to train the model.
        :param use_adaptive_fit: Boolean indicating whether the learning rate (alpha) should be altered
                                 as the cost value starts to stagnate.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X

        m, n = dataset.shape()[0]
        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(data, self.theta) + self.theta_zero

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, data)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # calculating cost in each iteration
            self.cost_history[i] = self.cost(dataset, scale)

            # stopping criteria (version 1)
            if i > 0:
                if np.abs(self.cost_history[i] - self.cost_history[i-1]) < 1:
                    if use_adaptive_fit:
                        self.alpha /= 2
                    else:
                        break

        return self



    def predict(self, dataset: Dataset, scale:bool = True) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        if scale:
            data = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            data = dataset.X

        return np.dot(data, self.theta) + self.theta_zero


    def score(self, dataset: Dataset, scale:bool = True) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        :param dataset: The dataset to compute the MSE on
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        y_pred = self.predict(dataset, scale)
        return mse(dataset.y, y_pred)


    def cost(self, dataset: Dataset, scale:bool = True) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        :param dataset: The dataset to compute the cost function on
        :param scale: Boolean indicating whether the data should be scaled (True) or not (False).
        """
        y_pred = self.predict(dataset, scale)
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))



if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegression()
    model.fit(dataset_)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score = model.score(dataset_)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f"Predictions: {y_pred_}")
