from si.data.dataset import Dataset
import numpy as np
from si.metrics.accuracy import accuracy

class StackingClassifier:
    """
    Uses a list of classification model instances to generate predictions. Those predictions are
    then used to train a final model and assist the model in generating its predictions.
    """
    def __init__(self, models, final_model):
        """
        Initializes the class.

        Parameters
        ----------
        :param models: A list of classification model instances
        :param final_model: A classification model to train and predict output values
                            using the outputs given by the chosen models.
        """
        self.models = models
        self.final_model = final_model



    def fit(self, dataset:Dataset) -> 'StackingClassifier':
        """
        Fits the chosen models using the given dataset and uses their output predictions
        to help fit the final model.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to train each model.
        """
        dataset_copy = Dataset(dataset.X, dataset.y, dataset.features, dataset.label)
        for m in self.models:
            m.fit(dataset)
            dataset_copy.X = np.c_[dataset_copy.X, m.predict(dataset)]

        self.final_model.fit(dataset_copy)
        return self



    def predict(self, dataset:Dataset):
        """
        Generates predictions with the chosen models and uses their outputs to
        help the chosen final model make its predictions. Returns those final predictions

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        dataset_copy = Dataset(dataset.X, dataset.y, dataset.features, dataset.label)
        for m in self.models:
            dataset_copy.X = np.c_[dataset_copy.X, m.predict(dataset)]

        return self.final_model.predict(dataset_copy)



    def score(self, dataset:Dataset):
        """
        Returns the accuracy score between the true and predicted output values.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class to predict the dependent variable.
        """
        return accuracy(dataset.y, self.predict(dataset))

