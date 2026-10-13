
import numpy as np
from sklearn.model_selection import train_test_split


# Split the dataset into training, validation, and testing sets
def split_dataset(X, y, test_size=0.2,
                  val_size=0.1, random_state=42):

    # Convert labels into a NumPy array
    y = np.asarray(y)

    # Check if the dataset split ratios are valid
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")

    if not (0 < val_size < 1 - test_size):
        raise ValueError(
            "val_size must be greater than 0 "
            "and test_size + val_size must be less than 1"
        )

    # Step 1: Split the dataset into Training + Validation and Testing
    # Use stratify to maintain the class distribution in each dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Step 2: Calculate the validation ratio from the remaining dataset
    val_ratio = val_size / (1 - test_size)

    # Split the remaining dataset into Training and Validation
    # Use stratify to maintain the class distribution
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train
    )

    # Return the training, validation, and testing datasets with their labels
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
