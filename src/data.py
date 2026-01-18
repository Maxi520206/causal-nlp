# src/data.py
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_imdb_splits(test_size: float = 0.2, seed: int = 42):
    ds = load_dataset("imdb")

    X = ds["train"]["text"]
    y = ds["train"]["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    X_test = ds["test"]["text"]
    y_test = ds["test"]["label"]

    return X_train, y_train, X_val, y_val, X_test, y_test
