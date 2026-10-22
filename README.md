 HEAD
# Logistic Regression from Scratch

A machine learning project implementing Logistic Regression from scratch using Python and NumPy, with a comparison against Scikit-learn's implementation.

## Project Overview

This project demonstrates how Logistic Regression works internally, including the sigmoid function, binary cross-entropy loss, gradient descent, and model evaluation.

## Dataset

- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** Scikit-learn
- **Task:** Binary classification
- **Features:** 30 numerical features
- **Classes:** Malignant and Benign

> This project is for educational purposes and is not intended for medical diagnosis.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pytest

## Implementation

The model is implemented from scratch with:

- Sigmoid activation function
- Binary cross-entropy loss
- Gradient descent optimization
- Prediction probabilities and class predictions
- Data preprocessing and feature scaling
- Model evaluation and comparison

Scikit-learn is used for dataset loading, preprocessing, evaluation metrics, and comparison with its built-in Logistic Regression model.

## Results

| Model | Test Accuracy | Test Log Loss |

| Logistic Regression from Scratch | 97.37% | 0.1078 |
| Scikit-learn Logistic Regression | 98.25% | 0.0777 |

*Results are from the current experiment and may vary if the configuration or data split changes.*

# Project Structure


Logistic-regression-from-scratch/
├── src/
│   ├── sigmoid.py
│   ├── log_loss.py
│   ├── gradient_descent.py
│   ├── logistic_regression.py
│   ├── data_loader.py
│   └── preprocessing.py
├── tests/
├── train.py
├── evaluate.py
├── compare_models.py
├── loss_curve.png
├── requirements.txt
├── .gitignore
└── README.md


# Run Locally

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

pip install -r requirements.txt

4. Train the model:

python train.py

5. Evaluate the model:


python evaluate.py


6. Compare the scratch implementation with Scikit-learn:

python compare_models.py


7. Run tests:


python -m pytest -v


##Learning Objectives

- Understand Logistic Regression mathematically and practically.
- Implement gradient descent using NumPy.
- Learn binary classification evaluation metrics.
- Compare a custom implementation with a standard ML library.

## Future Improvements

- Experiment with learning rates and iteration counts.
- Add regularization.
- Improve automated test coverage.
- Experiment with additional datasets.

# Logistic-regression-from-scratch
A from-scratch implementation of Logistic Regression using Numpy, with gradient descent, automated tests , and comparison against scikit-learn

