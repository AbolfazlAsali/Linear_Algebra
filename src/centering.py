import numpy as np
from pathlib import Path

def center_data(X):
    
    mean = X.mean(axis=0)
    B = X - mean

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    np.save(data_path / "centered_data.npy", B)

    return B, mean
