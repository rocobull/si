# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:18:33 2022

@author: rober
"""

import sys
sys.path.append("C:/Users/rober/si/src/si/data")
    
from dataset import Dataset
import numpy as np
from typing import Callable


class SelectPercentile:
    """
    Class that filters dataset variables based on their F-scores. Selects all variables
    with F-score values above the specified corresponding percentile.
    """
    
    def __init__(self, score_func:Callable[[object], tuple], percentile:int):
        """
        Stores the input values.
        
        Paramaters
        ----------
        :param score_func: f_classification() or f_regression() functions.
        :param percentile: Percentile value cut-off. Only F-scores above this
                           value will remain in the filtered dataset.
        """
        self.score_func = score_func
        self.perc = percentile
        
        
        
    def fit(self, dataset:object):
        """
        Stores the F-scores and respective p-values of each variable of the given dataset.
        
        Paramaters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    
    
    def transform(self, dataset:object) -> object:
        """
        Returns a filtered version of the given Dataset instance using their
        F-scores. The new dataset will have only the variables with F-scores above
        the specified percentile value.
        
        Paramaters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        inds = np.argsort(self.F)[::-1]
        ord_vals = np.sort(self.F)[::-1]
        perc_vals = np.percentile(ord_vals, self.perc)
        
        inds = inds[:sum(ord_vals <= perc_vals)]
        if dataset.features:
            features = np.array(dataset.features)[inds]
        else:
            features = None
            
        return Dataset(dataset.X[:, inds], dataset.y, features, dataset.label)
    
    
    
    def fit_transform(self, dataset:object) -> object:
        """
        Calls the fit() and transform() methods, returning the filtered version
        of the given Dataset instance.
        
        Paramaters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        model = self.fit(dataset)
        return model.transform(dataset)
    
    
    
    