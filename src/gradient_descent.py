import numpy as np
from src.sigmoid import sigmoid


def compute_gradients(X, y, weights, bias):
    """
    Compute gradients of binary logistic regression.

    Parameters
    ----------
    X : numpy.ndarray
        Input features with shape (n_samples, n_features).

    y : numpy.ndarray
        Actual binary labels with shape (n_samples,).

    weights : numpy.ndarray
        Model weights with shape (n_features,).

    bias : float
        Model bias.

    Returns
    -------
    tuple
        Gradients for weights and bias.
    """
    n_samples = X.shape[0]

    # Calculate prediction scores
    z = np.dot(X, weights) + bias

    # Convert scores into probabilities
    predictions = sigmoid(z)

    # Calculate prediction errors
    errors = predictions - y

    # Calculate gradients
    dw = np.dot(X.T, errors) / n_samples
    db = np.mean(errors)

    return dw, db