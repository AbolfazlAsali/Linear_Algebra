import numpy as np
from utils import load_data
from visualization import show_sample
from centering import center_data


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


if __name__ == "__main__":
    main()
