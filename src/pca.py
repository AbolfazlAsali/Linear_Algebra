import numpy as np
from pathlib import Path

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

def project_to_k_components(B, eigenvectors, k=10):
   
    W = eigenvectors[:, :k]
    T = B @ W 
    return T, W

# if __name__ == "__main__":

#     project_root = Path(__file__).resolve().parent.parent
#     data_dir = project_root / "data"

#     B = np.load(data_dir / "centered_data.npy")
#     eigenvalues = np.load(data_dir / "eigenvalues.npy")
#     eigenvectors = np.load(data_dir / "eigenvectors.npy")

#     k = 10
#     T, W = project_to_k_components(B, eigenvectors, k)

#     print(f"Original data shape: {B.shape}")
#     print(f"Projection matrix W shape: {W.shape}")
#     print(f"Transformed data T shape: {T.shape}")
#     print()

#     WtW = W.T @ W
#     is_orthonormal = np.allclose(WtW, np.eye(k))
#     print(f"W columns orthonormal: {is_orthonormal}")
#     print()

#     total_variance = eigenvalues.sum()
#     variance_k = eigenvalues[:k].sum()
#     explained_ratio = variance_k / total_variance
#     print(f"Variance explained by {k} components: {explained_ratio:.4f} ({explained_ratio*100:.2f}%)")
#     print()

#     np.save(data_dir / "transformed_data_k10.npy", T)
#     np.save(data_dir / "projection_matrix_W.npy", W)
#     print(f"Saved T to {data_dir / 'transformed_data_k10.npy'}")
#     print(f"Saved W to {data_dir / 'projection_matrix_W.npy'}")
