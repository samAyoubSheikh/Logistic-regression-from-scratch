
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import prepare_data
from src.logistic_regression import LogisticRegressionScratch


# 1. Prepare data
X_train, X_test, y_train, y_test = prepare_data()

# 2. Train our scratch-built model
model = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iterations=1000
)
model.fit(X_train, y_train)

# 3. Predict on unseen test data
y_pred = model.predict(X_test)

# 4. Calculate metrics
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, y_pred, zero_division=0), 4))

# 5. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 6. Detailed report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Malignant", "Benign"],
        zero_division=0
    )
)
