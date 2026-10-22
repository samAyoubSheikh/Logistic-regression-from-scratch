import numpy as np

def log_loss(y_true,y_pred):
    """
    Calculate binary cross-entropy loss between true labels and predicted probabilities.

    Parameters:

    y_true (numpy.ndarray): True binary labels (0 or 1).

    y_pred (numpy.ndarray): Predicted probabilities between 0 and 1.

    Returns:

    float
    Average binary log loss
    """

    y_true = np.asarray(y_true , dtype=float)
    y_pred = np.asarray(y_pred , dtype=float)

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
          + (1 - y_true) * np.log(1 - y_pred)
          )

    return float(loss)