# ADD "data" folder to modules path to be imported

#import sys
#sys.path.append("C:/Users/rober/si/src/si/data")



import numpy as np
import pandas as pd
from dataset import Dataset
from typing import Union

def read_csv(filename:str, sep:str = ",", features:bool = None, label:Union[None,int] = None) -> object:
    """
    Reads a csv file containing a dataframe and returns a Dataset object.
    
    Parameters
    ----------
    :param filename: The path of the desired csv file.
    :param sep: The string value that seperates each column.
    :param features: Boolean indicating if the feature names are present or not in the file
    :param label: The index of the column to be used as the dependent variable (y). Has the value "None" if this is not the case.
    """
    if features:
        col = 0
    else:
        col = False
    data = pd.read_csv(filename, sep = sep, index_col = col) #??????????
    
    data = data.to_numpy()
    if label == None:
        y = None
    else:
        y = data[:,label]
        data = np.delete(data, label, axis=1)
    #print(Dataset(data, y, features, label))
    return Dataset(data, y, features, label)


def write_csv(filename:str, dataset:object, sep:str = ",", features:bool = None, label:bool = None):
    """
    Saves the chosen dataset to a csv file.
    
    Parameters
    ----------
    :param filename: The path and name of the desired csv file to save the dataset.
    :param dataset: A Dataframe object.
    :param sep: The string value that will seperate each column in the csv file.
    :param features: Boolean indicating if the feature names are present or not in the file
    :param label: ...
    """
    dataset.to_csv(filename, sep) #ERRADO



if __name__ == "__main__":
    data = read_csv("C:/Users/rober/si/datasets/iris/iris.csv", label = 4)
    print(data)
    #write_csv("TEMP",data)
