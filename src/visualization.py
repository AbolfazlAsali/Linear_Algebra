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
