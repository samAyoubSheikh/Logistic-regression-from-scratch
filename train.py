
import matplotlib.pyplot as plt
from src.preprocessing import prepare_data
from src.logistic_regression import LogisticRegressionScratch
from src.log_loss import log_loss

# 1. Load and prepare data
X_train, X_test, y_train, y_test = prepare_data()

# 2. Create our own model
model = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iterations=2000
)

# 3. Train the model
model.fit(X_train, y_train)

plt.plot(model.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# 4. Display training information
print("Model trained successfully!")
print("Training samples:", X_train.shape[0])
print("Number of features:", X_train.shape[1])
print("Initial loss:", model.loss_history[0])
print("Final loss:", model.loss_history[-1])

# 5. Predict on unseen test data
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)

print("\nFirst 10 predicted classes:")
print(predictions[:10])

print("\nFirst 10 predicted probabilities:")
print(probabilities[:10])

print("\nTest-set log loss:", log_loss(y_test, probabilities))
