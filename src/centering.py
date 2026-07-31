import numpy as np

def center_data(X):
    mean = X.mean(axis=0)
    return X - mean, mean
