import numpy as np

def compute_covariance(B):
    m = B.shape[0]
    return (B.T @ B) / m

def eigen_decompose(C):
    
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]
