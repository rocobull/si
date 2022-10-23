# ADD "data" folder to modules path to be imported

import sys
sys.path.append("C:/Users/rober/si/src/si/data")

import numpy as np
import pandas as pd
from dataset import Dataset
from typing import Union



def read_csv(filename:str, sep:str = ",", features:bool = True, label:Union[None,int] = None) -> object:
    """
    Reads a csv file containing a dataframe and returns a Dataset object.
    
    Parameters
    ----------
    :param filename: The path of the desired csv file.
    :param sep: The string value that seperates each column.
    :param features: Boolean indicating if the feature names are present or not in the file
                     (It is assumed that feature names are present in the first column of the file).
    :param label: The index of the column to be used as the dependent variable (y). Has the value "None" if this is not the case.
                  TIP: Use -1 to select last column.
    """
    if features:
        row = 0
    else:
        row = None
    
    data = pd.read_csv(filename, sep = sep, header = row)
    
    #Get columns names:
    y_title = False
    feat = None
    
    if label:
        if features:
            feat = [elem for ix,elem in enumerate(data.columns) if ix != label]
            y_title = data.columns[label]
    else:
        if features:
            feat = list(data.columns)          
        
    #print(feat)
    
    
    #Seperate data (X and y variables) if needed:
    data = data.to_numpy()
    if not label:
        y = None
    else:
        y = data[:,label]
        data = np.delete(data, label, axis=1)
        
    return Dataset(data, y, feat, y_title)


def write_csv(filename:str, dataset:object, sep:str = ",", features:bool = True, label:bool = False):
    """
    Saves the chosen dataset to a csv file.
    
    Parameters
    ----------
    :param filename: The path and name of the desired csv file to save the dataset.
    :param dataset: A Dataframe object.
    :param sep: The string value that will seperate each column in the csv file.
    :param features: Boolean indicating if the feature names are present or not in the file
    :param label: Boolean indicating if the dependent variable (in case it exists) should be saved along with the other variables.
                  Will be saved at the last column
    """
    if not label:
        temp_array = np.column_stack((dataset.X, dataset.y))
    else:
        temp_array = dataset.X
    
    if features:
        cols = dataset.features
        header = True
    else:
        cols = None
        header = False
        
    to_save = pd.DataFrame(temp_array, columns=cols)
        
    to_save.to_csv(filename, sep, header=header, index=False)




if __name__ == "__main__":
    data = read_csv("C:/Users/rober/si/datasets/iris/iris.csv", label = 4)
    print(data.features)
    write_csv("TEMPORARIO.csv", data, features=True)
    
    data = read_csv("C:/Users/rober/si/datasets/iris/iris.csv", features=False, label = 4)
    write_csv("TEMPORARIO2.csv", data, features=True)
    #write_csv("TEMP",data)
