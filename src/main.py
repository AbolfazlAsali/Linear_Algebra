import numpy as np
from pathlib import Path
from utils import load_data
from centering import center_data
from visualization import show_sample
from pca import project_to_k_components
from visualization import plot_2d_scatter
from pca import compute_covariance, eigen_decompose
from visualization import show_sample, plot_cumulative_variance
from pca import compute_covariance, eigen_decompose, components_for_variance
from reconstruction import reconstruct_data, compute_reconstruction_error
from visualization import plot_reconstruction_error, plot_reconstruction_comparison



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
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "eigenvalues.npy", eigenvalues)
    np.save(data_dir / "eigenvectors.npy", eigenvectors)
    print(f"Top 5 eigenvalues: {eigenvalues[:5]}")
    print(f"Eigenvectors shape: {eigenvectors.shape}")
    print(f"Eigenvectors orthonormal: {np.allclose(eigenvectors.T @ eigenvectors, np.eye(64))}")



    k90, cum_var = components_for_variance(eigenvalues, threshold=0.90)
    print(f"\nComponents needed for 90% variance: {k90}")
    print(f"Fraction of original 64 dimensions: {k90/64:.2%}")
    print(f"Cumulative variance at k={k90}: {cum_var[k90-1]:.4f}")
    plot_cumulative_variance(cum_var)



    T10, W10 = project_to_k_components(B, eigenvectors, k=10)
    print(f"\nT10 shape: {T10.shape}")
    print(f"W10 shape: {W10.shape}")
    np.save(data_dir / "transformed_data_k10.npy", T10)
    np.save(data_dir / "projection_matrix_W.npy", W10)


    
    T2, _ = project_to_k_components(B, eigenvectors, k=2)
    plot_2d_scatter(T2, y, save_path="../pics/pca_2d_scatter.png")



       
    print("\n" + "=" * 60)
    print("Stage 9: Data Reconstruction and Error Analysis")
    print("=" * 60)
    
    k_values = [2, 10, 30]
    mse_values = []
    reconstructed_dict = {}
    
    for k in k_values:
        W = eigenvectors[:, :k]
        T = B @ W
        _, X_reconstructed = reconstruct_data(T, W, mean)
        
        mse = compute_reconstruction_error(X, X_reconstructed)
        mse_values.append(mse)
        reconstructed_dict[k] = X_reconstructed
        
        print(f"\nk={k}: MSE = {mse:.6f}")
        np.save(data_dir / f"reconstructed_data_k{k}.npy", X_reconstructed)
    
    
    plot_reconstruction_error(k_values, mse_values, save_path="../pics/reconstruction_error.png")
    
    plot_reconstruction_comparison(X, reconstructed_dict, sample_idx=0, 
                                  save_path="../pics/reconstruction_comparison_sample0.png")

    

if __name__ == "__main__":
    main()
