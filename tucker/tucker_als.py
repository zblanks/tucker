from typing import Tuple

import numpy as np

from tucker.hosvd import HOSVD


class ConvergenceWarning(Warning):
    pass


class TuckerALS:
    def __init__(
        self,
        X: np.ndarray,
        ranks: Tuple[int, ...],
        eta: float = 1e-7,
        max_iter: int = 100,
    ):

        self.X = X
        self.ranks = ranks
        self.eta = eta
        self.max_iter = max_iter

        assert len(X.shape) == 3, "Can only work with 3-way tensors"

        # Define empty matrices for the U, V, W, and G0 matrices
        self.U = np.empty(shape=(self.X.shape[0], self.ranks[0]))
        self.V = np.empty(shape=(self.X.shape[1], self.ranks[1]))
        self.W = np.empty(shape=(self.X.shape[2], self.ranks[2]))
        self.G0 = np.empty(shape=(self.ranks[2] * self.ranks[1], self.ranks[0]))

    def _initialize(self):
        self.hosvd = HOSVD(self.X, self.ranks)
        self.U, self.V, self.W, self.G0 = self.hosvd.fit()

    def _compute_left_singular_vector(self, mode: int) -> np.ndarray:
        X_mode = self.hosvd._unfold(mode)

        if mode == 0:
            M = X_mode @ np.kron(self.V, self.W)
        elif mode == 1:
            M = X_mode @ np.kron(self.U, self.W)
        else:
            M = X_mode @ np.kron(self.U, self.V)

        A, _, _ = np.linalg.svd(M)
        return A[:, : self.ranks[mode]]

    def _compute_objective(self):
        M = self.U.T @ self.hosvd._unfold(0) @ np.kron(self.V, self.W)
        return np.linalg.norm(M, ord="fro") ** 2

    def _run_als(self):
        self._losses = []
        self._losses.append(self._compute_objective())

        for _ in range(self.max_iter):
            self.U = self._compute_left_singular_vector(0)
            self.V = self._compute_left_singular_vector(1)
            self.W = self._compute_left_singular_vector(2)
            self._losses.append(self._compute_objective())

            # Check for ALS convergence
            if abs(self._losses[-1] - self._losses[-2]) <= self.eta:
                self.G0 = self.U.T @ self.hosvd._unfold(0) @ np.kron(self.V, self.W)
                return None

        # Failed to converge after max_iter
        raise ConvergenceWarning(
            f"Failed to converge after {self.max_iter} iterations."
        )

    def fit(self) -> Tuple[np.ndarray, ...]:
        self._initialize()
        self._run_als()
        return self.U, self.V, self.W, self.G0
