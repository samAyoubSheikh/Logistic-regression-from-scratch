import numpy as np

from src.sigmoid import sigmoid
from src.log_loss import log_loss
from src.gradient_descent import compute_gradients


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        # Gradient Descent
        for _ in range(self.n_iterations):
            probabilities = sigmoid(np.dot(X, self.weights) + self.bias)

            loss = log_loss(y, probabilities)
            self.loss_history.append(loss)

            dw, db = compute_gradients(
                X, y, self.weights, self.bias
            )

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.dot(X, self.weights) + self.bias
        return sigmoid(scores)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)