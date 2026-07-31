import numpy as np
from utils import load_data
from centering import center_data
from visualization import show_sample
from pca import compute_covariance, eigen_decompose
from visualization import show_sample, plot_cumulative_variance
from pca import compute_covariance, eigen_decompose, components_for_variance



def main():
    
    X, y = load_data()

    print(f"X shape: {X.shape}")   # (1797, 64)
    print(f"y shape: {y.shape}")   # (1797,)


    show_sample(X, y, index=0, save_path="../pics/sample_digit.png")


    B, mean = center_data(X)
    print(f"B shape: {B.shape}")
    print(f"Column means ≈ 0: {np.allclose(B.mean(axis=0), 0)}")
    print(f"Max absolute column mean: {np.abs(B.mean(axis=0)).max():.2e}")
    print(f"Mean vector (first 5): {mean[:5]}")



    C = compute_covariance(B)
    print(f"\nC shape: {C.shape}")
    print(f"C is symmetric: {np.allclose(C, C.T)}")
    print(f"C is PSD (all eigenvalues >= 0): {np.all(np.linalg.eigvalsh(C) >= -1e-10)}")



    eigenvalues, eigenvectors = eigen_decompose(C)
    print(f"Top 5 eigenvalues: {eigenvalues[:5]}")
    print(f"Eigenvectors shape: {eigenvectors.shape}")
    print(f"Eigenvectors orthonormal: {np.allclose(eigenvectors.T @ eigenvectors, np.eye(64))}")



    k90, cum_var = components_for_variance(eigenvalues, threshold=0.90)
    print(f"\nComponents needed for 90% variance: {k90}")
    print(f"Fraction of original 64 dimensions: {k90/64:.2%}")
    print(f"Cumulative variance at k={k90}: {cum_var[k90-1]:.4f}")

    plot_cumulative_variance(cum_var)

if __name__ == "__main__":
    main()
