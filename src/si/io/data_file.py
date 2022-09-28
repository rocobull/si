import numpy as np
from typing import Union
from dataset import Dataset

def read_data_file(filename, sep:str = ",", label:Union[None,int] = None):
    """
    Reads a txt file containing a dataframe and returns a Dataset object.
    
    Parameters
    ----------
    :param filename: The path of the desired txt file.
    :param sep: The string value that seperates each column.
    :param label: The index of the column to be used as the dependent variable (y). Has the value "None" if this is not the case.
    """
    data = np.genfromtxt(filename, delimiter=sep)
    if label == None:
        y = None
    else:
        y = data[:,label]
        data = np.delete(data, label, axis=1)
    #print(Dataset(data, y, features, label))
    return Dataset(data, y, label)
    

def write_data_file(filename:str, dataset:object, sep:str = ",", label:bool = None):
    """
    Saves the chosen dataset to a txt file.
    
    Parameters
    ----------
    :param filename: The path and name of the desired txt file to save the dataset.
    :param dataset: A Dataframe object.
    :param sep: The string value that will seperate each column in the csv file.
    :param label: ...
    """
    np.savetxt(filename, dataset, delimiter=sep) #ERRADO