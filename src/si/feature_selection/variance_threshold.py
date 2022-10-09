# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:57:58 2022

@author: rober
"""

import sys
sys.path.append("C:/Users/rober/si/src/si/data")

from dataset import Dataset

class VarianceThreshold:
    """
    Class that filters dataset variables based on their variance values.
    """
    
    def __init__(self, threshold:float):
        """
        Stores the input values.
        
        Paramaters
        ----------
        :param threshold: Limit value for the calculated F scores.
                          Any column with an F score below the threshold
                          will be removed from the Dataset object returned in
                          the transform() and fit_transform() methods.
        """
        self.thresh = threshold
        self.variance = None
        
        
    
    def fit(self, dataset:object):
        """
        Stores the variance values for each variable of the given dataset.
        
        Paramaters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        self.variance = dataset.get_var()
        return self
    
    
    
    def transform(self, dataset:object) -> object:
        """
        Returns a filtered version of the given Dataset instance using the
        specified threshold value.
        
        Paramaters
        ----------
        :param dataset: An instance of the Dataset class.
        """
        mask = self.variance > self.thresh
        new_X = dataset.X[:,mask]
        
        if not (dataset.features is None):
            dataset.features = [elem for ix,elem in enumerate(dataset.features) if mask[ix]]
            print(dataset.features)
            
        return Dataset(new_X, dataset.y, dataset.features, dataset.label)
    
    
    
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
    

    
if __name__ == "__main__":
    import numpy as np
    
    dataset = Dataset(np.array([[0, 2, 0, 3],
                                [1, 4, 2, 5],
                                [1, 2, 0, 1],
                                [0, 3, 0, 2]]),
                      np.array([1,2,3,4]),
                      ["1","2","3","4"], "5")
                      
    temp = VarianceThreshold(1)
    print(temp.fit_transform(dataset))