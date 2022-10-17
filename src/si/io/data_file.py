#import sys
#sys.path.append("C:/Users/rober/si/src/si/data")

import numpy as np
import pandas as pd
from typing import Union
from si.data.dataset import Dataset

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
    

def write_data_file(filename:str, dataset:object, sep:str = ",", label:bool = False):
    """
    Saves the chosen dataset to a txt file.
    
    Parameters
    ----------
    :param filename: The path and name of the desired txt file to save the dataset.
    :param dataset: A Dataframe object.
    :param sep: The string value that will seperate each column in the csv file.
    :param label: Boolean indicating if the dependent variable (in case it exists) should be saved along with the other variables.
                  Will be saved at the last column
    """
    if not label:
        temp_array = np.column_stack((dataset.X, dataset.y))
    else:
        temp_array = dataset.X
        
    to_save = pd.DataFrame(temp_array)
    np.savetxt(filename, to_save, delimiter=sep)


if __name__ == "__main__":
    pass





