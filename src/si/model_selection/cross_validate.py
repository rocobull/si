from collections.abc import Callable
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split


def cross_validate(model:object, dataset:Dataset, scoring:Callable = None, cv:int = 5, test_size:float = 0.3) -> dict:
    """
    Implements a cross validation algorithm to generate different test and train splits of the data
    for model fitting and prediction, returning a dictionary with the seed used, training data predictions score and
    test data prediction score for each iteration.

    Parameters
    ----------
    :param model: A model instance
    :param dataset: A dataset to use for training and testing the model
    :param scoring: An alternative scoring function to use
    :param cv: Number of iterations for cross validation
    :param test_size: Percentage of the data to be used as test data (value from 0 to 1)
    """
    scores = {"seed":[], "train":[], "test":[]}

    for i in range(cv):
        seed = np.random.randint(0, 1000)
        scores["seed"].append(seed)
        train, test = train_test_split(dataset, test_size, seed)
        model.fit(train)

        if not scoring:
            scores["train"].append(model.score(train))
            scores["test"].append(model.score(test))

        else:
            scores["train"].append(scoring(train.y, model.predict(train)))
            scores["test"].append(scoring(test.y, model.predict(test)))

    return scores
