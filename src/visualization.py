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
