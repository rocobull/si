# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:43:10 2022

@author: rober
"""
#!pip install numpy pandas scipy matplotlib

import sys
sys.path.append("C:/Users/rober/si/src/si/io")

import numpy as np
import pandas as pd
from typing import Union
#from data_file import read_data_file


class Dataset:
    """
    Creates a dataset and gives data distribution information. 
    """
    
    def __init__(self, X:np.ndarray, y:np.ndarray=None, features:Union[None,list]=None, label:Union[None,str]=None):
        """
        Stores the input values.
        
        Parameters
        ----------
        :param X: An independent variable matrix (should be a numpy.ndarray instance).
        :param X: The dependent variable vector (should be a numpy.ndarray instance).
        :param features: A vector constaining the names of each independent variable.
        :param label: The name of the dependent variable.
        """
        self.X = X
        self.y = y
        self.features = features
        self.label = label
    
    
    def __str__(self):
        result = "X:\n--\n"
        for elem in self.X:
            result += str(elem)[1:-1].replace(" ","\t") + "\n"
        
        if not (self.y is None):
            result += "\ny:\n--\n"
            for elem in self.y:
                result += str(elem) + "\n"
        
        return result
    
    #def __iter__(self):
    #    return pd.DataFrame(self)
    
    #def __getitem__(self, i):
    #    return self.X[i]
    
    
    def shape(self) -> tuple:
        """
        Returns the dimensions of both the independent and dependent variables.
        """
        return self.X.shape, self.y.shape
    
        
    def has_label(self) -> bool:
        """
        Checks to see if the dependent variable is available.
        """
        if self.y is None:
            return False
        else:
            return True
    
    
    
    def get_classes(self) -> Union[np.ndarray,None]:
        """
        Returns the unique values of the dependent variable.
        """
        if self.y is None:
            return None
        return np.unique(list(self.y), axis=0)
    
    
    
    def get_mean(self) -> np.ndarray:
        """
        Returns the mean value of each variable.
        """
        return np.mean(self.X, axis=0)
    
    
    
    def get_var(self) -> np.ndarray:
        """
        Returns the variance of each variable.
        """
        return np.var(self.X, axis=0)
    
    
    
    def get_median(self) -> np.ndarray:
        """
        Returns the median of each variable.
        """
        return np.median(self.X, axis=0)
    
    
    
    def get_min(self) -> np.ndarray:
        """
        Returns the minimum value of each variable.
        """
        return np.min(self.X, axis=0)



    def get_max(self) -> np.ndarray:
        """
        Returns the maximum value of each variable.
        """
        return np.max(self.X, axis=0)



    def summary(self) -> pd.DataFrame:
        """
        Returns a dataframe containing the mean, median, variance,
        minimum and maximum value of each variable.
        """
        return pd.DataFrame(
            {"mean": self.get_mean(),
            "median": self.get_median(),
            "var": self.get_var(),
            "min": self.get_min(),
            "max": self.get_max()}
        )
    
    
    
    def _find_nan(self, indiv:bool=True) -> list:
        """
        Returns a list of boolean values relative to the positions of missing values.
        
        Parameters
        ----------
        :param indiv: Boolean indicating if the list should contain boolean values
                      for each individual position ('True' if missing value was found),
                      or for each line ('True' if the line does not contain missing values).
        """
        df = pd.DataFrame(self.X)
        if not (self.y is None):
            df_y = pd.DataFrame(self.y)
            df = pd.concat([df, df_y], axis=1)
        if indiv:
            return pd.isnull(df)
        else:
            return ~pd.isnull(df).any(axis=1)
        
        
    
    def remove_nan(self, copy:bool=True):
        """
        Removes all rows containing missing values in either the dependent or
        independent variables.
        
        Parameters
        ----------
        :param copy: Boolean indicating whether the changes should be made
                     in a copy of the original data (True) or be done in-place
                     (False).
        """
        #NaN
        #---      
        cond = self._find_nan(False)
        new_X = self.X[cond,:]
        
        if not (self.y is None):
            new_y = self.y[cond]
            if copy:
                return new_X, new_y
            else:
                self.X = new_X
                self.y = new_y
        else:
            if copy:
                return new_X
            else:
                self.X = new_X
        
        
    
    
    def fill_nan(self, value = 0, copy:bool = True) -> Union[tuple,None]:
        """
        Replaces missing values with the value of the user's choice.
        
        Parameters
        ----------
        :param value: The value used to replace the missing values
        :param copy: Boolean indicating whether the changes should be made
                     in a copy of the original data (True) or be done in-place
                     (False).
        """
        cond = self._find_nan(indiv = True)
        if not (self.y is None):
            cond_y = cond.iloc[:,-1]
            cond = cond.iloc[:,0:-1]
            
        if copy:
            new_X = self.X.copy()
            new_X[cond] = value
            if not (self.y is None):
                new_y = self.y.copy()
                new_y[cond_y] = value
                return new_X, new_y
            else:
                return new_X
        else:
            self.X[cond] = value
            if not (self.y is None):
                self.y[cond_y] = value
    
    
    

if __name__ == "__main__":
    
    import CSV
    
    X = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [9,10,11,12]]) #4r, 4c
    
    y = np.array([10,
                  20,
                  30,
                  10]) #4r, 1c
    
    temp = Dataset(X,y)
    print(temp.shape())
    print(temp.has_label())
    print(temp.get_classes())
    print(temp.summary())
    print(temp.get_var())
    
    #With NAs (REMOVE)
    temp = CSV.read_csv("C:/Users/rober/si/datasets/iris/iris_missing_data.csv", label = 4)
    print(temp.shape())
    X,y = temp.remove_nan()
    print(X.shape, y.shape)
    
    
    #With NAs (FILL)
    temp = CSV.read_csv("C:/Users/rober/si/datasets/iris/iris_missing_data.csv", label = 4)
    X,y = temp.fill_nan(0)
    print(X)
    print(y)
    
    
    
    
    