from collections.abc import Callable
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def randomized_search_cv(model:object, dataset:Dataset, parameter_grid:dict = {}, scoring:Callable = None, cv:int = 5, n_iter:int = 10, test_size:float = 0.3):
    """
    Function to test multiple parameter values of the model to use, according to the number of iterations defined.
    Combinations are randomized in each iteration. Returns a dictionary with the seed used, training data
    predictions score and test data prediction score for each iteration, and the respective parameter values used.

    Parameters
    ----------
    :param model: A model instance
    :param dataset: A dataset to use for training and testing the model
    :param parameter_grid: Dictionary containing the model's parameters as keys and a list of the
                           respective values to test as values
    :param scoring: An alternative scoring function to use
    :param cv: Number of iterations for cross-validation
    :param n_iter: Number of parameter combinations to use
    :param test_size: Percentage of the data to be used as test data (value from 0 to 1)
    """
    for param in parameter_grid.keys():
        assert hasattr(model, param), f"'{param}' not a parameter of the chosen model."

    scores = []

    for _ in range(n_iter):
        parameters = {k: np.random.choice(v) for k,v in parameter_grid.items()}

        for param, val in parameters.items():
            setattr(model, param, val)

        score = cross_validate(model, dataset, scoring, cv, test_size)
        score["parameters"] = parameters

        scores.append(score)

    return scores



