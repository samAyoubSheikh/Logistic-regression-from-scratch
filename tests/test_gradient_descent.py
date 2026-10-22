import numpy as np
from src.gradient_descent import compute_gradients


def test_compute_gradients():
    X = np.array([[1.0], [2.0]])
    y = np.array([0.0, 1.0])
    weights = np.array([0.0])
    bias = 0.0

    dw, db = compute_gradients(X, y, weights, bias)

    assert np.allclose(dw, [-0.25])
    assert np.isclose(db, 0.0)