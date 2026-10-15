
import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from vgg_model import train_model, predict_model, VGG16, VGG_SMALL
from evaluate import evaluate_model, plot_history


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Planets and Moons")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Dataset settings
IMG_SIZE = 50
TEST_SIZE = 0.2
VAL_SIZE = 0.2
MAX_PER_CLASS = 3000
BATCH_SIZE = 64

# DCNN configurations
EXPERIMENTS = [
    ("VGG_SMALL_10", VGG_SMALL, 10),
    ("VGG_SMALL_20", VGG_SMALL, 20),
    ("VGG16_10", VGG16, 10),
    ("VGG16_20", VGG16, 20),
]


def main():

    print("-" * 60)
    print("DCNN Image Classification: Planets and Moons")
    print("-" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")

    images, labels, classes = load_data(
        DATA_PATH, IMG_SIZE, MAX_PER_CLASS
    )

    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

    with open(
        os.path.join(OUTPUT_DIR, "classes.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(classes, f, indent=4)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocess Images
    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    y = labels

    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)

    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    # Save training, validation, and testing datasets
    datasets = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    for name, data in datasets.items():
        np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), data)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 4: Train DCNN Models
    print("\n[Step 4] Training DCNN models...")

    results = []

    for name, blocks, epochs in EXPERIMENTS:

        print(f"\nTraining: {name}")
        print(f"Epochs: {epochs}")

        # Create output folder for each experiment
        experiment_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Train the model
        model, history = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            len(classes),
            experiment_dir,
            epochs,
            BATCH_SIZE,
            blocks
        )

        print(f"Training completed: {name}")

        # Step 5: Predict Test Dataset
        print(f"\n[Step 5] Predicting: {name}")

        predictions = predict_model(model, X_test)

        # Step 6: Evaluate Model
        print(f"\n[Step 6] Evaluating: {name}")

        evaluate_model(
            y_test,
            predictions,
            classes,
            save_path=os.path.join(
                experiment_dir, "confusion_matrix.png"
            )
        )

        # Plot training and validation accuracy/loss
        plot_history(
            history,
            os.path.join(
                experiment_dir, "training_history.png"
            )
        )

        # Calculate test accuracy
        accuracy = np.mean(
            np.asarray(predictions).reshape(-1) ==
            np.asarray(y_test).reshape(-1)
        )

        results.append({
            "Model": name,
            "Epochs": epochs,
            "Accuracy": float(accuracy)
        })

        print(f"Test Accuracy: {accuracy:.4f}")

    # Step 7: Compare DCNN Configurations
    print("\n[Step 7] Comparing model performance...")

    print("-" * 60)
    print(f"{'Model':<20} {'Epochs':<10} {'Accuracy':<10}")
    print("-" * 60)

    for result in results:
        print(
            f"{result['Model']:<20} "
            f"{result['Epochs']:<10} "
            f"{result['Accuracy']:.4f}"
        )

    # Save experiment results
    results_path = os.path.join(
        OUTPUT_DIR, "experiment_results.json"
    )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nAll experiments completed.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
