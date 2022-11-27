import numpy as np
from si.statistics.sigmoid_function import sigmoid_function


class Dense:

    def __init__(self, input_size, output_size):
        self.input_size =  input_size #Same size as attributes of input data
        self.output_size = output_size

        #Can be any initial values!
        self.weights = np.random.randn(input_size, output_size) #(lines, columns) (normal distribution between 0 e 1)
        self.bias = np.zeros((1, self.output_size))

    def forward(self, input_data:np.array): #Forward propagation
        """

        Parameters
        ----------
        :param X: Input data matrix (examples x attributes)
        """
        return np.dot(input_data, self.weights) + self.bias #Multiplies input data lines (examples) with weights columns, sums values of each column, and adds bias line




class SigmoidActivation:

    def __init__(self):
        pass

    def forward(self, input_data: np.array):
        return sigmoid_function(input_data)



class SoftMaxActivation:

    def __init__(self):
        pass

    def forward(self, input_data: np.array):
        ez = np.exp(input_data)
        return ez/(np.sum(ez, keepdims=True))



class ReLUActivation:

    def __init__(self):
        pass

    def forward(self, input_data: np.array):
        return np.maximum(input_data, 0)




