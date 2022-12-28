from collections.abc import Callable
import itertools

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate
from si.model_selection.split import train_test_split

from si.neural_network.nn import NN
from si.neural_network.layers import Dense, SigmoidActivation, SoftMaxActivation, ReLUActivation



def grid_search_cv(model:object,
                   dataset:Dataset,
                   parameter_grid:dict = {},
                   scoring:Callable = None,
                   cv:int = 5,
                   test_size:float = 0.3,
                   scale:bool = True,
                   verbose:bool = True,
                   starting_score:float = 0):
    """
    Function to test multiple parameter values of the model to use. Uses all possible combinations of the parameters.
    Returns a dictionary with the all calculated scores (seed used, training data predictions score and test data
    prediction score for each iteration), the grid of parameters that returns the best test scores,
    and the best trained model.

    Parameters
    ----------
    :param model: A model instance
    :param dataset: A dataset to use for training and testing the model
    :param parameter_grid: Dictionary containing the model's parameters as keys and a list of the
                           respective values to test as values
    :param scoring: An alternative scoring function to use
    :param cv: Number of iterations for cross-validation
    :param test_size: Percentage of the data to be used as test data (value from 0 to 1)
    :param scale: Boolean indicating whether the data should be scaled (True) or not (False)
    :param verbose: Indicates whether the score values for each grid combination should be printed (True) or not (False)
    :param starting_score: The starting value to consider as the best test score
    """
    for param in parameter_grid.keys():
        assert hasattr(model, param), f"'{param}' not a parameter of the chosen model."

    scores = []
    best_test_score = starting_score
    best_seed = 0
    best_model = None
    best_grid = []
    all_combs = itertools.product(*parameter_grid.values())

    for combs in all_combs:
        parameters = {}
        for param, val in zip(parameter_grid.keys(), combs):
            setattr(model, param, val)
            parameters[param] = val

        score = cross_validate(model, dataset, scoring, cv, test_size, scale)
        score["parameters"] = parameters

        check = _get_best_model(score, best_test_score)
        if check:
            best_seed, best_test_score = check
            best_grid = [{"seed": best_seed, "best_test_score": best_test_score}, parameters]
            train, test = train_test_split(dataset, test_size, best_seed)
            model.fit(train, scale=scale)
            best_model = model
        #print(score)

        scores.append(score)

    if verbose:
        _print_grid(scores, False)

    return scores, best_grid, best_model




def _get_best_model(scores:dict, best_score:float):
    """
    Determine the best model by comparing the maximum CV test_score with the current best score.

    Parameters
    ----------
    :param scores: Scores dictionary
    :param best_score: The current best score
    """
    best = False #If no score beats the current best score, returns False
    for ix,elem in enumerate(scores["test"]):
        if elem > best_score:
            best = [scores["seed"][ix], elem] #Return the best seed and test score

    return best




def _print_grid(scores:list, nn:bool = False):
    """
    Prints the scores of each combination in a readable manner

    Parameters
    ----------
    :param scores: Scores dictionary
    :param nn: Boolean value indicating if the 'grid_search_nn' was used (True) or the 'grid_search_cv' was used (False)
    """

    if nn:
        for dic in scores:
            print("Parameters of each layer:", dic["layers_grid"])
            print("|")
            for elem in dic["nn_grid"]:
                print(elem)
            print()

    else:
        for dic in scores:
            print()
            print("Params:\t\t\t", dic["parameters"])
            print("Seeds:\t\t\t", dic["seed"])
            print("Train scores:\t", dic["train"])
            print("Test scores:\t", dic["test"])






########## Grid search for neural network models: ##########



def _verif(dataset, layers, layers_parameter_grids):
    """
    Auxiliary functions to check if 'layers' and 'parameter_grids' lists are valid.
    Returns list of layer objects corresponding the list of strings ('layers') given.

    Parameters
    ----------
    :param dataset: A dataset to use for training and testing the model
    :param layers: Names (strings) of the layers to use (should be "dense", "sigmoid", "softmax" or "relu")
    :param layers_parameter_grids: List of parameter dictionaries (one for each layer)
    """
    assert len(layers) == len(layers_parameter_grids), "Length of 'layers' should be equal to length of 'parameter_grids'."

    for layer in [l.lower() for l in set(layers)]:
        assert layer in ("dense", "sigmoid", "softmax", "relu"), \
            f'Invalid layer type: {layer}.' \
            f'Should be "dense", "sigmoid", "softmax" or "relu"'

    real_layers = []
    for layer in layers:
        if layer.lower() == "dense":
            # len_output will be set to a random int initially (in this case 50).
            # This value should be optimized
            real_layers.append(Dense(len(dataset.X[0]), 66))
        elif layer.lower() == "sigmoid":
            real_layers.append(SigmoidActivation())
        elif layer.lower() == "softmax":
            real_layers.append(SoftMaxActivation())
        else:
            real_layers.append(ReLUActivation())

    for ix in range(len(layers_parameter_grids)):
        for param in layers_parameter_grids[ix].keys():
            assert hasattr(real_layers[ix], param), f"'{param}' not a parameter of the chosen model."

    return real_layers



def _get_full_param_grid(layers_parameter_grids):
    """
    Returns a list of dictionaries for every possible combination of parameters for every layer

    Parameters
    ----------
    :param layers_parameter_grids: List of parameter dictionaries (one for each layer)
    """
    all_param_grid = {}
    attr_names = [] #Names of attributes to optimize (list of len(layers) lists)
    for ix, grid in enumerate(layers_parameter_grids):
        attr_names.append(list(grid.keys()))
        all_param_grid[ix] = [elem for elem in itertools.product(*grid.values())]

    return attr_names, [elem for elem in itertools.product(*all_param_grid.values())]




def grid_search_nn(dataset:Dataset,
                   all_layers:list = [],
                   layers_param_grids:list = [],
                   nn_param_grid:dict = {},
                   scoring:Callable = None,
                   cv:int = 5,
                   test_size:float = 0.3,
                   scale:bool = True,
                   verbose:bool = True):
    """
    Parameters
    ----------
    :param dataset: A dataset to use for training and testing the model
    :param all_layers: A list of layers (strings) to compile onto the neural network model ("dense", "sigmoid", "softmax", "relu")
    :param nn_param_grid: The parameters pertaining the neural network ('NN') object.
                          'layers' parameter should not be included here
    :param scoring: An alternative scoring function to use
    :param cv: Number of iterations for cross-validation
    :param test_size: Percentage of the data to be used as test data (value from 0 to 1)
    :param scale: Boolean indicating whether the data should be scaled (True) or not (False)
    :param verbose: Indicates whether the score values for each grid combination should be printed (True) or not (False)
    """

    true_layers = _verif(dataset, all_layers, layers_param_grids) #Returns layer instances
    attr_names, layers_full_grids = _get_full_param_grid(layers_param_grids)
    #attr_names: List of lists, each containing the names of the attributes to optimize per layer (in order)
    #layers_full_grids: List of tuples combinations

    #print(attr_names)
    #print(layers_full_grids)

    scores = []
    parameters = {}
    best_test_score = 0
    best_model = None
    best_grid = {}

    for grid in layers_full_grids:
        for ix, layer_params in enumerate(grid):
            parameters[ix] = parameters.get(ix, []) + [{k:v for k,v in zip(attr_names[ix], layer_params)}]

    #'parameters' keys are the index of each layer in the NN, and their values are the
    #respecitve layers' hyperparameters in order.

    for grid in range(len(layers_full_grids)):   #Grid combination indices
        grid_history = {"nn_grid":None, "layers_grid":{}}

        for layer in range(len(all_layers)): #Layer indices
            grid_history["layers_grid"][layer] = {"name": all_layers[layer]}

            for param_name in attr_names[layer]:  #Parameter names for each layer
                new_param = parameters[layer][grid][param_name]
                setattr(true_layers[layer], param_name, new_param)
                grid_history["layers_grid"][layer][param_name] = new_param

        grid_history["nn_grid"], best_nn_grid, temp_model = grid_search_cv(NN(true_layers),dataset, nn_param_grid,
                                                                           scoring, cv, test_size, scale, verbose=False,
                                                                           starting_score=best_test_score)

        if not (temp_model is None): #If a model has been created that returns a test score better than the current one
            best_grid["seed"] = best_nn_grid[0]["seed"]
            best_grid["best_test_score"] = best_nn_grid[0]["best_test_score"]
            best_grid["nn_grid"] = best_nn_grid[1]
            best_grid["layers_grid"] = grid_history["layers_grid"]
            best_model = temp_model
            best_test_score = best_nn_grid[0]["best_test_score"]

        scores.append(grid_history)

    if verbose:
        _print_grid(scores, True)

    return scores, best_grid, best_model
