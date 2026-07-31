import numpy as np
from sklearn.datasets import load_digits


def load_data():

    digits = load_digits()
    X = digits.data      # shape: (1797, 64)
    y = digits.target    # shape: (1797,)
    return X, y
