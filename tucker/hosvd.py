from typing import Tuple

import numpy as np


class HOSVD:
    """Implements Higher-Order SVD for 3-way tensors

    Parameters
    ----------
    X : np.ndarray
        3-way tensor
    ranks : Tuple[int, ...]
        U, V, W, G0 decomposition of X
    """

    def __init__(self, X: np.ndarray, ranks: Tuple[int, ...]) -> None:
        self.X = X
        self.ranks = ranks

        assert len(self.X.shape) == 3, "Can only work with 3-way tensors"

    def _unfold(self, mode: int) -> np.ndarray:
        return np.reshape(np.moveaxis(self.X, mode, 0), (self.X.shape[mode], -1))

    def _compute_left_singular_vectors(self, mode: int, rank: int) -> np.ndarray:
        X_mode = self._unfold(mode)
        U, _, _ = np.linalg.svd(X_mode)
        return U[:, :rank]

    def fit(self) -> Tuple[np.ndarray, ...]:
        """Fits the HOSVD; currently assumes only 3-way tensor

        Returns
        -------
        Tuple[np.ndarray, ...]
            Matrices plus core tensor in matrix form
        """
        U = self._compute_left_singular_vectors(0, self.ranks[0])
        V = self._compute_left_singular_vectors(1, self.ranks[1])
        W = self._compute_left_singular_vectors(2, self.ranks[2])
        G0 = U.T @ self._unfold(0) @ np.kron(V, W)
        return U, V, W, G0
