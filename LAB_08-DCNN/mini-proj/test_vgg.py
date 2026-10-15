
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Test settings
MODEL_NAME = "VGG_SMALL_20"
N_SAMPLES = 4


def test_vgg(n_samples=N_SAMPLES, model_name=MODEL_NAME):

    # Set model and output paths
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    model_path = os.path.join(model_dir, "vgg_model.keras")

    # Check model file
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # Load model and test dataset
    model = keras.models.load_model(model_path)

    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    with open(
        os.path.join(OUTPUT_DIR, "classes.json"),
        encoding="utf-8"
    ) as f:
        classes = json.load(f)

    # Validate sample count
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    n_samples = min(n_samples, len(X_test))

    # Select random test images
    indices = np.random.choice(
        len(X_test),
        n_samples,
        replace=False
    )

    X_sample = X_test[indices]
    y_sample = y_test[indices]

    # Predict selected images
    probabilities = model.predict(X_sample, verbose=0)

    if probabilities.shape[-1] == 1:

        probabilities = probabilities.ravel()
        predictions = (probabilities > 0.5).astype(int)

        confidence = np.where(
            predictions == 1,
            probabilities,
            1 - probabilities
        )

    else:

        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    # Create prediction figure
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.4 * cols, 4.0 * rows)
    )

    axes = np.atleast_1d(axes).ravel()

    # Display prediction results
    for i, ax in enumerate(axes):

        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[int(predictions[i])]
        true = classes[int(y_sample[i])]

        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            f"Pred: {pred} ({confidence[i] * 100:.1f}%)\n"
            f"True: {true}",
            color=color
        )

        print(
            f"[{i + 1}] "
            f"Pred: {pred:<12} "
            f"True: {true:<12} "
            f"Confidence: {confidence[i] * 100:.1f}% "
            f"{'OK' if correct else 'WRONG'}"
        )

    # Calculate sample accuracy
    correct_total = int((predictions == y_sample).sum())

    print(f"\nModel: {model_name}")
    print(f"Correct: {correct_total}/{n_samples}")

    # Save prediction figure
    fig.suptitle(
        f"{model_name}: {correct_total}/{n_samples} correct"
    )

    fig.tight_layout()

    save_path = os.path.join(
        model_dir,
        "prediction_sample.png"
    )

    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_vgg()
