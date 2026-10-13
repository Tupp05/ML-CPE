
import json
import os
import csv  # Save model comparison results

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model

# Import functions for comparison graphs and prediction visualization
from evaluate import (
    evaluate_model,
    plot_history,
    plot_comparison,
    plot_predictions
)


# ==============================
# Configuration
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the dataset directory
DATA_PATH = os.path.join(
    BASE_DIR,
    "Planets_Moons_Data",
    "Planets and Moons"
)

# Define the outputs directory
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Define configuration parameters
IMG_SIZE = 100  # Image size: 100x100 pixels
TEST_SIZE = 0.2  # Use 20% of the dataset for testing
VAL_SIZE = 0.1  # Use 10% of the dataset for validation
MAX_PER_CLASS = 3000  # Limit each class to 3000 images
BATCH_SIZE = 32  # Number of images processed per batch

# Set the random seed for reproducibility
RANDOM_STATE = 42

# Define four CNN experiments with different configurations
EXPERIMENTS = [
    {
        "name": "Model_A",
        "conv_layers": 2,
        "dense_neurons": 128,
        "epochs": 10
    },
    {
        "name": "Model_B",
        "conv_layers": 2,
        "dense_neurons": 128,
        "epochs": 20
    },
    {
        "name": "Model_C",
        "conv_layers": 3,
        "dense_neurons": 128,
        "epochs": 10
    },
    {
        "name": "Model_D",
        "conv_layers": 3,
        "dense_neurons": 128,
        "epochs": 20
    }
]


def main():

    print("=" * 60)
    print("CNN Image Recognition: Planets and Moons")
    print("=" * 60)

    # Set the random seed for the experiments
    from tensorflow import keras
    keras.utils.set_random_seed(RANDOM_STATE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==============================
    # Step 1: Load Dataset
    # ==============================

    print("\n[Step 1] Loading dataset...")

    # Pass dataset path, image size, and class limit to load_data()
    # in data_loader.py to load images and labels
    images, labels, classes = load_data(
        DATA_PATH,
        IMG_SIZE,
        MAX_PER_CLASS
    )

    # Save labels and class names to the outputs directory
    np.save(
        os.path.join(OUTPUT_DIR, "labels.npy"),
        labels
    )

    with open(
        os.path.join(OUTPUT_DIR, "classes.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(classes, f, indent=4)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # ==============================
    # Step 2: Preprocessing
    # ==============================

    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)  # Convert images into a contiguous NumPy array using preprocessing.py
    y = labels

    # Save the processed image features to the outputs directory
    np.save(
        os.path.join(OUTPUT_DIR, "features.npy"),
        X
    )

    print(f"Feature shape: {X.shape}")

    # ==============================
    # Step 3: Split Dataset
    # ==============================

    print("\n[Step 3] Splitting dataset...")

    # Pass image features, labels, split ratios, and random state
    # to split_dataset() in split_data.py
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X,
        y,
        TEST_SIZE,
        VAL_SIZE,
        random_state=RANDOM_STATE
    )

    # Save split datasets
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)

    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # ==============================
    # Step 4: Train CNN Models
    # ==============================

    # Store the evaluation results of each model
    results = []

    for exp in EXPERIMENTS:

        print("\n" + "=" * 60)
        print(f"Training: {exp['name']}")
        print(f"Conv Layers   : {exp['conv_layers']}")
        print(f"Dense Neurons : {exp['dense_neurons']}")
        print(f"Epochs        : {exp['epochs']}")
        print("=" * 60)

        # Reset the random seed before building each model
        keras.utils.set_random_seed(RANDOM_STATE)

        # Create a separate output directory for each model
        model_dir = os.path.join(
            OUTPUT_DIR,
            exp["name"]
        )

        os.makedirs(model_dir, exist_ok=True)

        # Train the CNN model
        model, history = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            len(classes),
            output_dir=model_dir,
            epochs=exp["epochs"],
            batch_size=BATCH_SIZE,
            conv_layers=exp["conv_layers"],
            dense_neurons=exp["dense_neurons"],
            use_early_stopping=False
        )

        print(f"\n{exp['name']} Training completed.")

        # ==============================
        # Step 5: Prediction
        # ==============================

        print("\n[Step 5] Testing model...")

        # Predict labels for the test dataset
        predictions = predict_model(
            model,
            X_test
        )

        # Calculate test accuracy
        test_accuracy = float(
            np.mean(predictions == y_test)
        )

        print(
            f"{exp['name']} Test Accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        # ==============================
        # Step 6: Evaluation
        # ==============================

        print("\n[Step 6] Evaluating model...")

        # Generate and save the confusion matrix
        evaluate_model(
            y_test,
            predictions,
            classes,
            save_path=os.path.join(
                model_dir,
                "confusion_matrix.png"
            )
        )

        # Plot training and validation accuracy and loss
        plot_history(
            history,
            os.path.join(
                model_dir,
                "training_history.png"
            )
        )

        # Display sample images with their predicted labels
        plot_predictions(
            model,
            X_test,
            y_test,
            classes,
            save_path=os.path.join(
                model_dir,
                "predictions.png"
            )
        )

        # Store the experiment results
        results.append({
            "model": exp["name"],
            "conv_layers": exp["conv_layers"],
            "dense_neurons": exp["dense_neurons"],
            "epochs": exp["epochs"],
            "train_accuracy": float(
                history.history["accuracy"][-1]
            ),
            "val_accuracy": float(
                history.history["val_accuracy"][-1]
            ),
            "test_accuracy": test_accuracy
        })

    # ==============================
    # Step 7: Compare CNN Models
    # ==============================

    print("\n[Step 7] Comparing CNN models...")

    # Save model comparison results to a CSV file
    csv_path = os.path.join(
        OUTPUT_DIR,
        "comparison_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    # Generate an accuracy comparison graph
    plot_comparison(
        results,
        save_path=os.path.join(
            OUTPUT_DIR,
            "accuracy_comparison.png"
        )
    )

    # ==============================
    # Step 8: Summary
    # ==============================

    print("\n" + "=" * 60)
    print("CNN EXPERIMENT RESULTS")
    print("=" * 60)

    # Compare convolutional layers, epochs, and accuracy
    for result in results:
        print(
            f"{result['model']} | "
            f"Conv: {result['conv_layers']} | "
            f"Epochs: {result['epochs']} | "
            f"Train: {result['train_accuracy'] * 100:.2f}% | "
            f"Val: {result['val_accuracy'] * 100:.2f}% | "
            f"Test: {result['test_accuracy'] * 100:.2f}%"
        )

    # Select the best model based on validation accuracy
    best_model = max(
        results,
        key=lambda r: r["val_accuracy"]
    )

    print("\nBest Model (by Validation Accuracy):")
    print(f"Model    : {best_model['model']}")
    print(f"Val Acc  : {best_model['val_accuracy'] * 100:.2f}%")
    print(f"Test Acc : {best_model['test_accuracy'] * 100:.2f}%")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
