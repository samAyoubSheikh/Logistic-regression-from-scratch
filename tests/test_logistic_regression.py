import numpy as np

from src.logistic_regression import LogisticRegressionScratch 


def test_model_trains_successfully():
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegressionScratch(
        learning_rate=0.1,
        n_iterations=2000
    )

    model.fit(X, y)

    assert model.weights is not None
    assert len(model.loss_history) == 2000


def test_predictions_are_binary():
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegressionScratch(
        learning_rate=0.1,
        n_iterations=2000
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert set(predictions).issubset({0, 1})


def test_probabilities_are_between_zero_and_one():
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegressionScratch(
        learning_rate=0.1,
        n_iterations=2000
    )

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_loss_decreases_during_training():
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegressionScratch(
        learning_rate=0.1,
        n_iterations=2000
    )

    model.fit(X, y)

    assert model.loss_history[-1] < model.loss_history[0]