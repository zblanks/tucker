import numpy as np

from tucker.hosvd import HOSVD
from tucker.tucker_als import TuckerALS


def test_orthogonal():
    rng = np.random.default_rng(17)
    X = rng.random(size=(3, 4, 2))

    hosvd = HOSVD(X, (2, 2, 2))
    U, V, W, _ = hosvd.fit()

    # Check that U, V, and W are orthogonal
    assert np.allclose(U.T @ U, np.eye(2))
    assert np.allclose(V.T @ V, np.eye(2))
    assert np.allclose(W.T @ W, np.eye(2))


def test_increasing_objective_function():
    rng = np.random.default_rng(17)
    X = rng.standard_normal(size=(2, 2, 2))
    tals = TuckerALS(X, (1, 1, 1))
    tals.fit()

    # Check for monotonic increase of the objective function
    n = len(tals._losses)
    for i in range(1, n):
        assert tals._losses[i] >= tals._losses[i - 1]
