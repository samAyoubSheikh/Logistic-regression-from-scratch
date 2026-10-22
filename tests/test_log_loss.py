import numpy as np

from src.log_loss import log_loss

def test_log_loss_for_correct_positive_prediction():
    loss = log_loss([1],[0.9])
    assert np.isclose(loss, 0.1053605, atol=1e-5)


def test_log_loss_for_correct_negative_prediction():
    loss = log_loss([0],[0.1])
    assert np.isclose(loss, 0.1053605, atol=1e-5)

def test_log_loss_for_uncertain_prediction():
    loss = log_loss([1],[0.5])
    assert np.isclose(loss, 0.693147, atol=1e-5)


def test_log_loss_for_multiple_prediction():
    loss = log_loss([1, 0], [0.9, 0.1])
    assert np.isclose(loss, 0.1053605, atol=1e-5)