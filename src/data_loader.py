from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_dataset():
    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(dataset.target, name="target")

    return X, y


if __name__ == "__main__":
    X, y = load_dataset()

    print("Dataset shape:", X.shape)
    print("\nFirst 5 rows:")
    print(X.head())

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget labels:")
    print(load_breast_cancer().target_names)