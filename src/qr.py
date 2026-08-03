import numpy as np

def qr_algorithm(C, n_iter=100):
   
    C_k = C.copy()
    for _ in range(n_iter):
        Q, R = np.linalg.qr(C_k, mode='reduced')
        C_k = R @ Q

    return C_k, Q, R


if __name__ == "__main__":
    
    seeds = [42, 123, 456]  
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Testing with seed = {seed}")
        print('='*60)
        
        np.random.seed(seed)
        M = np.random.rand(4, 4)
        A = M @ M.T

        print("Original symmetric matrix A:\n", A)
        print("Eigenvalues (true, via eigh):", np.linalg.eigvalsh(A))

        C_final, Q, R = qr_algorithm(A, n_iter=100)

        print("\nMatrix after QR algorithm (should be ~diagonal):\n", C_final)
        print("Diagonal entries (estimated eigenvalues):", np.diag(C_final))
        print("\nQ shape:", Q.shape, "| R shape:", R.shape)
        print("Columns of Q orthonormal:", np.allclose(Q.T @ Q, np.eye(Q.shape[1])))

