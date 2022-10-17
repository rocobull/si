import numpy as np

def euclidean_distance(x:np.array, y:np.array) -> np.array:
    return np.sqrt(((x-y)**2).sum(axis=1)) #Rows (axis=0 is for columns)