import logging
from pathlib import Path

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


def face_data_comparison(p: Path):
    X = scipy.io.loadmat(p / "FERETC80A45.mat")["fea2D"] / 255.0

    # Get the Tucker approximation using my version of the ALS algorithm
    t_als = TuckerALS(X, (5, 5, 15))
    U, V, W, G0 = t_als.fit()
    X0_hat = U @ G0 @ np.kron(V, W).T
    X_hat = tl.fold(X0_hat, 0, X.shape)
    relative_error = compute_relative_error(X, X_hat)
    logging.info(f"Faces relative error (my implementation): {relative_error}")

    # Get the TensorLy Tucker approximation for comparison
    G, factors = tucker(X, rank=[5, 5, 15])
    G0 = tl.unfold(G, mode=0)
    X0_hat = factors[0] @ G0 @ np.kron(factors[1], factors[2]).T
    X_hat = tl.fold(X0_hat, 0, X.shape)
    relative_error = compute_relative_error(X, X_hat)
    logging.info(f"Faces relative error (TensorLy version): {relative_error}")


def main():
    logging.basicConfig(filename="results.log", level=logging.DEBUG)
    p = Path("data")

    face_data_comparison(p)

    return None


if __name__ == "__main__":
    main()
