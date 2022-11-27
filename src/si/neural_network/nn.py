from si.neural_network.layers import Dense, SigmoidActivation
from si.data.dataset import Dataset

class NN:

    def __init__(self, layers:list=[]):
        self.layers = layers

    def fit(self, dataset:Dataset) -> 'NN':
        data = dataset.X
        for layer in self.layers:
            data = layer.forward(data)
        return data

