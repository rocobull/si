from si.neural_network.layers import Dense, SigmoidActivation
from si.data.dataset import Dataset
from si.metrics.mse import mse, mse_derivative
import numpy as np
from typing import Callable
from sklearn import preprocessing

class NN:
    """
    Creates an artificial neural network model
    """
    def __init__(self,
                 layers:list = [],
                 epochs:int = 1000,
                 learning_rate:float = 0.01,
                 loss:Callable = mse,
                 loss_derivative:Callable = mse_derivative,
                 verbose:bool = True):
        """
        Initializes the class' global attributes.

        Parameters
        ----------
        :param layers: A list of layer instances to be compiled in the network in order
        :param epochs: The number of iterations over the entire training dataset
        :param learning_rate: A float value to influence the weight values update
                              (smaller values update the weights at a slower rate)
        :param loss: The loss metric function to use for model evaluation
        :param loss_derivative: The derivative value of the loss function to be used
                                during the backpropagation algorithm.
        :param verbose: A boolean indicating if the loss value for each epoch
                        should be printed (True) or not (False)
        """
        self.epochs = int(epochs)
        assert self.epochs >= 1, "Number of epochs should be superior or equal to 1"

        self.layers = layers
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose

        self.history = {}

        #Redifine "input_size" of 'Dense' layers (to avoid dimensionality errors)
        prev_output = None
        for layer in layers:
            if layer.__class__.__name__ == 'Dense':
                if prev_output:
                    layer.input_size = prev_output
                    layer.weights = np.random.randn(prev_output, layer.output_size) #Redefine weights dimensions
                    prev_output = layer.output_size
                else:
                    prev_output = layer.output_size
                    layer.weights = np.random.randn(layer.input_size, layer.output_size)

                layer.bias = np.zeros((1, layer.output_size))




    def fit(self, dataset:Dataset, scale=True) -> 'NN':
        """
        Trains the neural network model through successive forward and backward passes.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class
        :param scale: Boolean indicating if the dataset should be scaled (True) or not (False)
        """
        if scale:
            X = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            X = dataset.X

        for epoch in range(1, self.epochs + 1):

            y_pred = np.array(X)
            y_true = np.reshape(dataset.y, (-1, 1))

            for layer in self.layers:
                y_pred = layer.forward(y_pred)

            #print(y_pred)
            #print("-----------------")
            #Return index of maximum value
            for ix,pred in enumerate(y_pred):
                #print(np.where(pred == max(pred))[0])
                y_pred[ix] = np.where(pred == max(pred))[0][0]
            #print(y_pred)
            #y_pred = np.apply_along_axis(lambda x: np.where(x == max(x))[0][0], axis=1, arr=y_pred)
            #   print(y_pred[0])

            error = self.loss_derivative(y_true, y_pred)
            for layer in self.layers[::-1]:
                error = layer.backward(error, self.learning_rate)

            #save history
            cost = self.loss(y_pred, y_true)
            self.history[epoch] = cost

            #print loss
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} = {cost}')

        return y_pred



    def predict(self, dataset:Dataset, scale=True):
        """
        Predicts the output values of a given dataset.

        Parameters
        ----------
        :param dataset: An instance of the Dataset class
        :param scale: Boolean indicating if the dataset should be scaled (True) or not (False)
        """
        if scale:
            X = preprocessing.scale(dataset.X, axis=0)  # Scale each feature
        else:
            X = dataset.X

        for layer in self.layers:
            X = layer.forward(X)

        return np.apply_along_axis(lambda x: np.where(x == max(x))[0][0], axis=1, arr=X)

