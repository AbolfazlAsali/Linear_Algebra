import numpy as np
from pathlib import Path

def reconstruct_data(T, W, mean):
   
    B_reconstructed = T @ W.T
    X_reconstructed = B_reconstructed + mean
    return B_reconstructed, X_reconstructed


def compute_reconstruction_error(X_original, X_reconstructed):
    mse = np.mean((X_original - X_reconstructed) ** 2)
    return mse
