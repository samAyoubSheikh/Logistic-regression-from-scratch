from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from src.preprocessing import prepare_data
from src.logistic_regression import LogisticRegressionScratch

# Prepare the same train/test split
X_train, X_test, y_train, y_test = prepare_data()

# Our scratch-built model
scratch_model = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iterations=2000
)

scratch_model.fit(X_train, y_train)

scratch_predictions = scratch_model.predict(X_test)
scratch_probabilities = scratch_model.predict_proba(X_test)

# Scikit-learn model
sklearn_model = LogisticRegression(max_iter=2000)
sklearn_model.fit(X_train, y_train)

sklearn_predictions = sklearn_model.predict(X_test)
sklearn_probabilities = sklearn_model.predict_proba(X_test)[:, 1]

# Compare performance
print("SCRATCH MODEL")
print("Accuracy:", round(
    accuracy_score(y_test, scratch_predictions), 4
))
print("Log Loss:", round(
    log_loss(y_test, scratch_probabilities), 4
))

print("\nSCIKIT-LEARN MODEL")
print("Accuracy:", round(
    accuracy_score(y_test, sklearn_predictions), 4
))
print("Log Loss:", round(
    log_loss(y_test, sklearn_probabilities), 4
))