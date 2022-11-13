from collections.abc import Callable
import itertools
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def grid_search_cv(model:object, dataset:Dataset, parameter_grid:dict = {}, scoring:Callable = None, cv:int = 5, test_size:float = 0.3):
    """
    Function to test multiple parameter values of the model to use. Uses all possible combinations of the parameters.
    Returns a dictionary with the seed used, training data predictions score and test data prediction score
    for each iteration, and the respective parameter values used.

    Parameters
    ----------
    :param model: A model instance
    :param dataset: A dataset to use for training and testing the model
    :param parameter_grid: Dictionary containing the model's parameters as keys and a list of the
                           respective values to test as values
    :param scoring: An alternative scoring function to use
    :param cv: Number of iterations for cross-validation
    :param test_size: Percentage of the data to be used as test data (value from 0 to 1)
    """
    for param in parameter_grid.keys():
        assert hasattr(model, param), f"'{param}' not a parameter of the chosen model."

    scores = []
    all_combs = itertools.product(*parameter_grid.values())

    for combs in all_combs:
        parameters = {}
        for param, val in zip(parameter_grid.keys(), combs):
            setattr(model, param, val)
            parameters[param] = val

        score = cross_validate(model, dataset, scoring, cv, test_size)
        score["parameters"] = parameters

        scores.append(score)

    return scores
