from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io
import tensorly as tl
from tensorly.decomposition import tucker

from tucker.tucker_als import TuckerALS


def compute_relative_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Computes the relative Tucker error using the true and approximated tensor

    Parameters
    ----------
    X : np.ndarray
        True tensor
    X_hat : np.ndarray
        Approximated tensor from Tucker decomposition

    Returns
    -------
    float
        Relative error
    """
    signal = 0.0
    error = 0.0

    for k in range(X.shape[2]):
        signal += np.linalg.norm(X[..., k], ord="fro") ** 2
        error += np.linalg.norm(X[..., k] - X_hat[..., k], ord="fro") ** 2

    return error / signal


def compute_tensor_approximation(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    G0: np.ndarray,
    orig_shape: Tuple[int, ...],
) -> np.ndarray:

    X0_hat = U @ G0 @ np.kron(V, W).T
    return tl.fold(X0_hat, 0, orig_shape)


def face_data_comparison(p: Path):
    X: np.ndarray = scipy.io.loadmat(p / "FERETC80A45.mat")["fea2D"] / 255.0

    # Get the Tucker approximation using my version of the ALS algorithm
    t_als = TuckerALS(X, (5, 5, 15))
    U, V, W, G0 = t_als.fit()
    X_hat = compute_tensor_approximation(U, V, W, G0, X.shape)
    relative_error = compute_relative_error(X, X_hat)
    print(f"Faces relative error (my implementation): {relative_error}")

    # Get the TensorLy Tucker approximation for comparison
    G, factors = tucker(X, rank=[5, 5, 15])
    G0 = tl.unfold(G, mode=0)
    X_hat = compute_tensor_approximation(
        factors[0], factors[1], factors[2], G0, X.shape
    )

    relative_error = compute_relative_error(X, X_hat)
    print(f"Faces relative error (TensorLy version): {relative_error}")


def generate_rank_options(num_choices: int, rng) -> List[Tuple[int, ...]]:
    # Assumes 3-way tensors
    r0 = rng.integers(low=1, high=15, size=num_choices)
    r1 = rng.integers(low=2, high=4, size=num_choices)
    r2 = rng.integers(low=2, high=4, size=num_choices)
    return list(zip(r0, r1, r2))


def compute_compression_ratio(
    tensor_shape: Tuple[int, ...], ranks: Tuple[int, ...]
) -> float:

    i, j, k = tensor_shape
    r0, r1, r2 = ranks
    return (i * j * k) / ((i * r0) + (j * r1) + (k * r2) + (r0 * r1 * r2))


def run_own_data_compression_experiment(X: np.ndarray, ranks: Tuple[int, ...]) -> None:
    t_als = TuckerALS(X, ranks, max_iter=200)
    U, V, W, G0 = t_als.fit()
    X_hat = compute_tensor_approximation(U, V, W, G0, X.shape)
    relative_error = compute_relative_error(X, X_hat)
    compression_ratio = compute_compression_ratio(X.shape, ranks)

    msg = f"Rank configuration: {ranks}; Relative Error: {relative_error}; Compression Ratio: {compression_ratio}"
    print(msg)


def compute_svd_approx_matrix_diff(X: np.ndarray, mode: int, rank: int) -> np.ndarray:
    X_mode = tl.unfold(X, mode)
    U, S, Vt = np.linalg.svd(X_mode)

    # Subset the factors based on the rank
    U = U[:, :rank]
    S = np.diag(S[:rank])
    Vt = Vt[:rank, :]

    X_hat = U @ S @ Vt
    return X_mode - X_hat


def compute_svd_error(X: np.ndarray, X_diff: np.ndarray) -> float:
    # First have to compute the "signal" for the original tensor
    signal = 0.0
    for k in range(X.shape[-1]):
        signal += np.linalg.norm(X[..., k], ord="fro") ** 2

    # Next compute the approximation error from the SVD for the unfolding
    error = np.linalg.norm(X_diff, ord="fro") ** 2
    return error / signal


def main():
    p = Path("data")

    face_data_comparison(p)

    # Personal data compression experiment; I'll try various random rank
    # tuples and log the results
    X: np.ndarray = np.load(p / "data.npy")
    run_own_data_compression_experiment(X, (5, 2, 3))

    """
    To now do the SVD approximation we have three unfoldings: X0 = (2000 x (4*10)),
    X1 = (4 x (2000*10)), and X2 = (10 x (2000*4)). Using the Tucker model,
    we're representing the tensor with 
    (5 * 2000) + (1 * 4) + (2 * 10) + (5 * 1 * 2) = 10,068 floats.
    So for the three unfoldings we need, X0: 2,040 * r -> r = 5;
    X1: 20,004 * r -> r = 1; X2: 8,010 * r -> r = 2 as the closest approximations
    at the given compression ratio
    """
    # X0 SVD approximation
    X_diff = compute_svd_approx_matrix_diff(X, 0, rank=5)
    relative_error = compute_svd_error(X, X_diff)
    print(f"Mode 0 Relative Error {relative_error}")

    # X1 SVD approximation
    X_diff = compute_svd_approx_matrix_diff(X, 1, rank=1)
    relative_error = compute_svd_error(X, X_diff)
    print(f"Mode 1 Relative Error {relative_error}")

    # X2 SVD approximation
    X_diff = compute_svd_approx_matrix_diff(X, 2, rank=2)
    relative_error = compute_svd_error(X, X_diff)
    print(f"Mode 2 Relative Error: {relative_error}")

    return None


if __name__ == "__main__":
    main()
