import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PICS_DIR = os.path.join(BASE_DIR, "..", "pics")


def show_sample(X, y, index=0, save_path=None):

    image = X[index].reshape(8, 8)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {y[index]}")
    plt.axis('off')

    if save_path is None:
        save_path = os.path.join(PICS_DIR, "sample_digit.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.show()


def plot_cumulative_variance(cum_var, save_path=None):

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=3)
    plt.axhline(0.90, color="r", linestyle="--", label="90% threshold")
    plt.xlabel("Number of components (k)")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative Explained Variance vs. k")
    plt.legend()
    plt.grid(True)

    if save_path is None:
        save_path = os.path.join(PICS_DIR, "cumulative_variance.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_2d_scatter(T2, labels, save_path=None):
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(T2[:, 0], T2[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label="Digit")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Projection of Digits")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PICS_DIR, "pca_2d_scatter.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_reconstruction_error(k_values, mse_values, save_path=None):

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mse_values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Number of Components (k)', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.title('Reconstruction Error vs Number of Principal Components', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(PICS_DIR, "reconstruction_error.png")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_reconstruction_comparison(original, reconstructed_dict, sample_idx=0, save_path=None):
    
    n_plots = 1 + len(reconstructed_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3))
    
    
    axes[0].imshow(original[sample_idx].reshape(8, 8), cmap='gray')
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
   
    for idx, (k, reconstructed) in enumerate(sorted(reconstructed_dict.items()), start=1):
        axes[idx].imshow(reconstructed[sample_idx].reshape(8, 8), cmap='gray')
        axes[idx].set_title(f'k={k}', fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(PICS_DIR, f"reconstruction_comparison_sample{sample_idx}.png")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()