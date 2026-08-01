import numpy as np
from sklearn.datasets import load_digits


def load_data():

    digits = load_digits()
    X = digits.data      
    y = digits.target    
    return X, y
