import numpy as np
from src.sigmoid import sigmoid

def test_sigmoid_at_zero():
    assert np.isclose(sigmoid(0), 0.5)


def test_sigmoid_at_positive_value():
    assert np.isclose(sigmoid(2), 0.880797,atol=1e-5)


def test_sigmoid_at_negative_value():
    assert np.isclose(sigmoid(-2), 0.1192029,atol=1e-5)


def test_sigmoid_array():
    test_values = np.array([-2, 0, 2])

    expected_probabilities = np.array([0.119203, 0.5, 0.880797])
    
    assert np.allclose(sigmoid(test_values), expected_probabilities, atol=1e-5)