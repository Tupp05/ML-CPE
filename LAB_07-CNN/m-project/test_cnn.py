
import json
import os
import csv  # Read model comparison results

import matplotlib

# Set Matplotlib backend to save graphs without displaying them
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Define the project and outputs directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Define the number of sample images for testing
N_SAMPLES = 4

# Select the CNN model to test
# "best" selects the model with the highest validation accuracy
# Or specify "Model_A", "Model_B", "Model_C", or "Model_D"
MODEL_NAME = "best"

# Set the random seed to select the same sample images each time
RANDOM_STATE = 42


# Find the saved CNN model based on the selected model name
def get_model_path(model_name):

    # Define the path of the model comparison results
    csv_path = os.path.join(
        OUTPUT_DIR, "comparison_results.csv"
    )

    # Check if the comparison results file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            "comparison_results.csv not found. "
            "Please run main.py first."
        )

    # Read the experiment results from the CSV file
    with open(csv_path, "r", encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    # Check if the CSV file contains experiment results
    if not results:
        raise ValueError("No experiment results found.")

    # Select the model with the highest validation accuracy
    if model_name == "best":
        selected = max(
            results,
            key=lambda r: float(r["val_accuracy"])
        )
    else:
        # Find the model that matches the specified model name
        selected = next(
            (
                r for r in results
                if r["model"] == model_name
            ),
            None
        )

        if selected is None:
            raise ValueError(
                f"Model '{model_name}' not found."
            )

    # Generate the model filename based on the configuration in cnn_model.py
    filename = (
        f"cnn_conv{selected['conv_layers']}"
        f"_dense{selected['dense_neurons']}"
        f"_epoch{selected['epochs']}.keras"
    )

    # Define the full path of the selected model
    model_path = os.path.join(
        OUTPUT_DIR,
        selected["model"],
        filename
    )

    # Check if the trained model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return model_path, selected["model"]


# Load the selected CNN model and test it with sample images
def test_cnn(n_samples=N_SAMPLES):

    # Check if the number of sample images is valid
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    # Get the model path and model name using get_model_path()
    model_path, selected_model = get_model_path(
        MODEL_NAME
    )

    print(f"\nSelected model: {selected_model}")
    print(f"Loading: {model_path}")

    # Load the trained CNN model from the saved .keras file
    model = keras.models.load_model(model_path)

    # Load the test images and labels from the outputs directory
    X_test = np.load(
        os.path.join(OUTPUT_DIR, "X_test.npy")
    )

    y_test = np.load(
        os.path.join(OUTPUT_DIR, "y_test.npy")
    )

    # Load class names from classes.json
    with open(
        os.path.join(OUTPUT_DIR, "classes.json"),
        "r",
        encoding="utf-8"
    ) as f:
        classes = json.load(f)

    # Limit the number of samples to the available test images
    n_samples = min(n_samples, len(X_test))

    # Create a random generator with a fixed seed
    rng = np.random.default_rng(RANDOM_STATE)

    # Randomly select test image indices without duplicates
    indices = rng.choice(
        len(X_test),
        size=n_samples,
        replace=False
    )

    # Get the selected test images and their actual labels
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    # Pass the selected images to the CNN model for prediction
    probabilities = model.predict(
        X_sample,
        verbose=0
    )

    # Convert prediction probabilities into class labels
    if probabilities.shape[-1] == 1:
        probabilities = probabilities.ravel()

        # Use a threshold of 0.5 for binary classification
        predictions = (
            probabilities > 0.5
        ).astype(int)

        # Calculate confidence based on the predicted class
        confidence = np.where(
            predictions == 1,
            probabilities,
            1 - probabilities
        )

    else:
        # Select the class with the highest probability for multiclass classification
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    # Calculate the number of rows and columns for displaying sample images
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))

    # Create a graph for displaying prediction results
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.8 * cols, 4.2 * rows)
    )

    axes = np.atleast_1d(axes).ravel()

    # Display each sample image with its predicted and actual class
    for i, ax in enumerate(axes):

        # Hide unused image positions
        if i >= n_samples:
            ax.axis("off")
            continue

        # Get the predicted and actual class names
        pred = classes[int(predictions[i])]
        true = classes[int(y_sample[i])]

        # Check if the predicted class matches the actual class
        correct = predictions[i] == y_sample[i]

        # Use green for correct predictions and red for incorrect predictions
        color = "green" if correct else "red"

        # Display the sample image without axis labels
        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])

        # Display predicted class, confidence, and actual class
        ax.set_title(
            f"Pred: {pred} ({confidence[i] * 100:.1f}%)\n"
            f"True: {true}",
            color=color,
            fontsize=10
        )

        # Print the prediction results in the terminal
        print(
            f"[{i + 1}] "
            f"Pred: {pred:<12} "
            f"True: {true:<12} "
            f"Confidence: {confidence[i] * 100:5.1f}% "
            f"{'OK' if correct else 'WRONG'}"
        )

    # Count the number of correctly predicted sample images
    correct_total = int(
        (predictions == y_sample).sum()
    )

    print(
        f"\nCorrect: {correct_total}/{n_samples}"
    )

    # Display the selected model name and number of correct predictions
    fig.suptitle(
        f"{selected_model}: "
        f"{correct_total}/{n_samples} correct"
    )

    fig.tight_layout()

    # Save prediction results in the selected model directory
    save_path = os.path.join(
        OUTPUT_DIR,
        selected_model,
        "prediction_sample.png"
    )

    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_cnn()
