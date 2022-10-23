# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:17:42 2022

@author: rober
"""
from scipy import stats


def f_classification(dataset:object) -> tuple:
    """
    Returns the F-scores and respective p-values of each variable in the given dataset.
    
    Paramaters
    ----------
    :param dataset: An instance of the Dataset class.
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p


if __name__ == "__main__":
    import sys
    sys.path.append("C:/Users/rober/si/src/si/data")
    
    from dataset import Dataset
    import numpy as np
    
    dataset = Dataset(np.array([[0, 2, 0, 3],
                                [1, 4, 2, 5],
                                [1, 2, 0, 1],
                                [0, 3, 0, 2],
                                [5, 5, 10, 6]]),
                      np.array([1,1,2,2,2]),
                      ["1","2","3","4"], "5")
    print(f_classification(dataset))

