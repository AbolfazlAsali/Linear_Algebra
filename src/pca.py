import numpy as np

def compute_covariance(B):
    m = B.shape[0]
    return (B.T @ B) / m

def eigen_decompose(C):
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]

def explained_variance_ratio(eigenvalues):
    total = eigenvalues.sum()
    return eigenvalues / total

def cumulative_variance(eigenvalues):
    ratio = explained_variance_ratio(eigenvalues)
    return np.cumsum(ratio)

def components_for_variance(eigenvalues, threshold=0.90):
    cum_var = cumulative_variance(eigenvalues)
    k = int(np.searchsorted(cum_var, threshold) + 1)
    return k, cum_var
