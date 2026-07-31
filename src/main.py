from utils import load_data
from visualization import show_sample


def main():
    
    X, y = load_data()

    print(f"X shape: {X.shape}")   # (1797, 64)
    print(f"y shape: {y.shape}")   # (1797,)

    show_sample(X, y, index=0, save_path="../pics/sample_digit.png")


if __name__ == "__main__":
    main()
