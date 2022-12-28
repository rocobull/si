import numpy as np
from si.statistics.sigmoid_function import sigmoid_function


class Dense:
    """
    A class to create fully connected layers to compile onto a
    Neural Network class ('NN') instance.
    """
    def __init__(self, input_size:int, output_size:int):
        """
        Initializes the class' global attributes.

        Parameters
        ----------
        :param input_size: The number of input values
        :param output_size: The number of output values
        """
        self.input_size =  input_size #Same size as attributes of input data
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size) #(lines, columns) (normal distribution between 0 e 1)
        self.bias = np.zeros((1, self.output_size))
        self.X = None


    def forward(self, input_data:np.array): #Forward propagation
        """
        Returns the dot product between the input data and the layer's weight values.
        The layer's bias is added to each resulting value.

        Parameters
        ----------
        :param input_data: The resulting values of a neural network layer (or input data)
        """
        self.X = input_data
        return np.dot(input_data, self.weights) + self.bias # Multiplies input data lines (examples) with weights
                                                            # columns, sums values of each column, and adds bias line

    def backward(self, error:np.ndarray, learning_rate:float = 0.001):
        """
        Applies the gradient descent algorithm to update the weights of the layer.

        Parameters
        ----------
        :param error: The value of the error function derivative
        :param learning_rate: A float value to influence the weight values update
                              (smaller values update the weights at a slower rate)
        """
        error_to_propagate = np.dot(error, self.weights.T)

        self.weights -= learning_rate*np.dot(self.X.T, error) #'error' é o valor do gradiente para a camada atual
        self.bias -= learning_rate*np.sum(error, axis=0)

        return error_to_propagate



class SigmoidActivation:
    """
    A class to create a sigmoid activation function to compile onto a
    Neural Network class ('NN') instance.
    """
    def __init__(self):
        self.X = None

    def forward(self, input_data: np.array):
        """
        Applies the sigmoid activation function to the input data
        (returns values between 0 and 1).

        Parameters
        ----------
        :param input_data: The resulting values of a neural network layer (or input data)
        """
        self.X = input_data
        return sigmoid_function(input_data)


    def backward(self, error:np.ndarray, learning_rate:bool = 0.001):
        """
        Returns the input error value

        Parameters
        ----------
        :param error: The value of the error function derivative
        :param learning_rate: A boolean value to influence the weight values update
                              (smaller values update the weights at a slower rate)
        """
        sigmoid_derivative = 1/(1+np.exp(-self.X))
        sigmoid_derivative = sigmoid_derivative * (1 - sigmoid_derivative)

        error_to_propagate = error * sigmoid_derivative

        return error_to_propagate



class SoftMaxActivation:
    """
    A class to create a SoftMax activation function to compile onto a
    Neural Network class ('NN') instance.
    """
    def __init__(self):
        pass

    def forward(self, input_data: np.array):
        """
        Applies the SoftMax activation function to the input data
        (returns values between 0 and 1).

        Parameters
        ----------
        :param input_data: The resulting values of a neural network layer (or input data)
        """
        ez = np.exp(input_data)
        return ez/(np.sum(ez, keepdims=True))

    def backward(self, error:np.ndarray, learning_rate:bool = 0.001):
        """
        Returns the input error value

        Parameters
        ----------
        :param error: The value of the error function derivative
        :param learning_rate: A boolean value to influence the weight values update
                              (smaller values update the weights at a slower rate)
        """
        return error



class ReLUActivation:
    """
    A class to create a ReLU activation function to compile onto a
    Neural Network class ('NN') instance.
    """
    def __init__(self):
        self.X = None

    def forward(self, input_data: np.array):
        """
        Applies the ReLU activation function to the input data
        (returns the maximum value between the input data and 0).

        Parameters
        ----------
        :param input_data: The resulting values of a neural network layer (or input data)
        """
        self.X = input_data
        return np.maximum(input_data, 0)


    def backward(self, error:np.ndarray, learning_rate:bool = 0.001):
        """
        Returns the input error value

        Parameters
        ----------
        :param error: The value of the error function derivative
        :param learning_rate: A boolean value to influence the weight values update
                              (smaller values update the weights at a slower rate)
        """
        error_to_propagate = np.where(self.X > 0, 1, 0)
        return error_to_propagate



