
import matplotlib

# Set Matplotlib backend before importing pyplot to save graphs without displaying them
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Evaluate model predictions and display accuracy, classification report, and confusion matrix
def evaluate_model(y_test, predictions, classes, save_path=None):

    # Create a list of class labels based on the number of classes
    labels = list(range(len(classes)))

    # Calculate accuracy by comparing actual labels with predicted labels
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    # Generate a classification report for all classes
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)
    print("Confusion Matrix:")

    # Generate a confusion matrix to compare actual and predicted classes
    matrix = confusion_matrix(
        y_test, predictions, labels=labels
    )
    print(matrix)

    # Save the confusion matrix graph if a save path is provided
    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


# Create and save a confusion matrix graph
def plot_confusion_matrix(matrix, classes, save_path):

    # Set the graph size to display all 11 classes clearly
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the confusion matrix using a blue color map
    im = ax.imshow(matrix, cmap="Blues")

    # Add a colorbar to represent the number of predictions
    fig.colorbar(im, ax=ax)

    # Set class names on the X-axis and rotate them to prevent overlapping
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")

    # Set class names on the Y-axis
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Calculate a threshold to adjust text color based on cell values
    threshold = matrix.max() / 2

    # Display the number of predictions inside each matrix cell
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, matrix[i, j],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[i, j] > threshold
                else "black"
            )

    # Adjust graph layout and save the image
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# Plot training and validation accuracy and loss for each epoch
def plot_history(history, save_path):
    """Plot training and validation accuracy/loss."""

    # Create two graphs for accuracy and loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate epoch numbers starting from 1
    epochs = range(
        1, len(history.history["accuracy"]) + 1
    )

    # Plot training accuracy
    axes[0].plot(
        epochs,
        history.history["accuracy"],
        label="Training"
    )

    # Plot validation accuracy
    axes[0].plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].legend()

    # Add a grid to the accuracy graph
    axes[0].grid(True, alpha=0.3)

    # Plot training loss
    axes[1].plot(
        epochs,
        history.history["loss"],
        label="Training"
    )

    # Plot validation loss
    axes[1].plot(
        epochs,
        history.history["val_loss"],
        label="Validation"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].legend()

    # Add a grid to the loss graph
    axes[1].grid(True, alpha=0.3)

    # Adjust graph layout and save the image
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")


# Compare training, validation, and testing accuracy across CNN models
def plot_comparison(results, save_path):

    # Get model names from the experiment results
    names = [r["model"] for r in results]

    # Convert training accuracy into percentages
    train_acc = [
        r["train_accuracy"] * 100 for r in results
    ]

    # Convert validation accuracy into percentages
    val_acc = [
        r["val_accuracy"] * 100 for r in results
    ]

    # Convert testing accuracy into percentages
    test_acc = [
        r["test_accuracy"] * 100 for r in results
    ]

    # Define bar positions and bar width
    x = np.arange(len(names))
    width = 0.25

    # Create a bar chart for comparing CNN models
    fig, ax = plt.subplots(figsize=(11, 6))

    # Display training accuracy bars
    ax.bar(
        x - width, train_acc,
        width, label="Training"
    )

    # Display validation accuracy bars
    ax.bar(
        x, val_acc,
        width, label="Validation"
    )

    # Display testing accuracy bars
    ax.bar(
        x + width, test_acc,
        width, label="Testing"
    )

    ax.set_xlabel("CNN Configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CNN Model Accuracy Comparison")

    # Set model names and accuracy range on the graph
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)

    # Add a legend and grid to the graph
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Adjust graph layout and save the comparison graph
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")


# Display sample test images with actual and predicted class names
def plot_predictions(model, X_test, y_test,
                     classes, save_path, num_images=12):

    # Set a random seed to select the same sample images each time
    rng = np.random.default_rng(42)

    # Limit the number of sample images to the available test images
    count = min(num_images, len(X_test))

    # Randomly select image indices without duplicates
    indices = rng.choice(
        len(X_test),
        size=count,
        replace=False
    )

    # Pass selected test images to the model for prediction
    probabilities = model.predict(
        X_test[indices], verbose=0
    )

    # Convert prediction probabilities into class labels
    if probabilities.shape[-1] == 1:
        predictions = (
            probabilities.ravel() > 0.5
        ).astype(int)
    else:
        predictions = np.argmax(probabilities, axis=1)

    # Create a 3x4 grid to display prediction examples
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    axes = axes.flatten()

    # Hide axes for all image positions
    for ax in axes:
        ax.axis("off")

    # Display each image with its actual and predicted class
    for i, idx in enumerate(indices):

        image = X_test[idx]
        true_label = int(y_test[idx])
        predicted_label = int(predictions[i])

        # Check if the predicted class matches the actual class
        correct = true_label == predicted_label

        # Display green text for correct predictions and red for incorrect predictions
        axes[i].imshow(image)
        axes[i].set_title(
            f"True: {classes[true_label]}\n"
            f"Pred: {classes[predicted_label]}",
            color="green" if correct else "red",
            fontsize=10
        )
        axes[i].axis("off")

    # Adjust graph layout and save the prediction examples
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")
