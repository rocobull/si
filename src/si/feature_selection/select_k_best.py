# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:29:45 2022

@author: rober
"""
#import sys
#sys.path.append("C:/Users/rober/si/src/si/data")
    
from si.data.dataset import Dataset
import numpy as np
from typing import Callable

class SelectKBest:
    """
    Class that filters dataset variables based on their F-scores. Selects only the
    top 'k' variables.
    """
    def __init__(self, score_func:Callable[[object], tuple], k:int):
        """
        Stores the input values.
        
        Parameters
        ----------
        :param score_func: f_classification() or f_regression() functions.
        :param k: Top 'k' variables to keep in the filtered dataset.
        """
        self.score_func = score_func
        self.k = k #Número de valores a devolver (dos melhores)
        self.F = None
        self.p = None
        
        
        
    def fit(self, dataset:object):
        """
        Stores the F-scores and respective p-values of each variable of the given dataset.
        
        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    
    
    def transform(self, dataset:object) -> object:
        """
        Returns a filtered version of the given Dataset instance using their
        F-scores. The new dataset will have only the top 'k' variables
        (with the largest F-scores).
        
        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        inds = np.argsort(self.F)[-self.k:][::-1] #ordem decrescente
        features = np.array(dataset.features)[inds]
        return Dataset(dataset.X[:, inds], dataset.y, features, dataset.label)
    
    
    
    def fit_transform(self, dataset:object) -> object:
        """
        Calls the fit() and transform() methods, returning the filtered version
        of the given Dataset instance.
        
        Parameters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        model = self.fit(dataset)
        return model.transform(dataset)
    
    
#THE BIGGER THE VALUE OF F, THE BIGGER THE DISTANCE BETWEEN THE DATA!

#Number of F statistics = number of independent variable columns

if __name__=="__main__":
    
    #sys.path.append("C:/Users/rober/si/src/si/statistics")
    from si.statistics.f_classification import f_classification
    
    dataset = Dataset(np.array([[0, 2, 0, 3, 10, 4, 2],
                                [1, 4, 2, 5, 19, 20, 0],
                                [1, 2, 0, 1, 12, 10, 3],
                                [0, 3, 0, 2, 14, 8, 8],
                                [5, 5, 10, 6, 11, 12, 10],
                                [2, 3, 6, 3, 17, 18, 5]]),
                      np.array([1,1,1,2,2,2]),
                      ["1","2","3","4","5","6","7"], "8")
    temp = SelectKBest(f_classification, 3)
    temp.fit(dataset)
    print(temp.transform(dataset))
    #print(temp.fit_transform(dataset))