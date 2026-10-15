
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.1):

    # Convert labels to NumPy array
    y = np.asarray(y)

    # Validate dataset sizes
    if len(X) != len(y):
        raise ValueError("Features and labels must have the same length.")

    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    if not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1.")

    if test_size + val_size >= 1:
        raise ValueError(
            "The sum of test_size and val_size must be less than 1."
        )

    # Split training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Calculate validation ratio
    val_ratio = val_size / (1 - test_size)

    # Split training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
