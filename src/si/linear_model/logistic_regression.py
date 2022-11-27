
import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function

class LogisticRegression:
    """
    A linear regression model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique

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

    def fit(self, dataset: Dataset, use_adaptive_fit: bool = False) -> 'LogisticRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to train the model.
        :param use_adaptive_fit: Boolean indicating whether the learning rate (alpha) should be altered
                                 as the cost value starts to stagnate.
        """
        m, n = dataset.shape()[0]

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        print(self.max_iter)
        for i in range(self.max_iter):
            # predicted y
            y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # calculating cost in each iteration
            self.cost_history[i] = self.cost(dataset)

            # stopping criteria (version 1)
            if i > 0:
                if np.abs(self.cost_history[i] - self.cost_history[i - 1]) < 0.0001:
                    if use_adaptive_fit:
                        self.alpha /= 2
                    else:
                        break

        return self



    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        pred_vals = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        mask = pred_vals >= 0.5
        pred_vals[mask] = 1
        pred_vals[~mask] = 0
        return pred_vals

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)



    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using regularization

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        pred_vals = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        m, n = dataset.shape()[0]
        y = dataset.y
        regularization = self.l2_penalty/(2*m)*np.sum(self.theta**2)
        #for elem in y:
        #    if elem == 1:
        #        cond.append(np.log(y_pred))
        #    else:
        #        cond.append(np.log(1-y_pred))
        cond = np.log(1-pred_vals)

        return -1/m * np.sum(y*np.log(pred_vals) + (1-y)*cond) + regularization


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0,1,0,0])
    dataset_ = Dataset(X=X, y=y)
    print(dataset_)

    # fit the model
    model = LogisticRegression()
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