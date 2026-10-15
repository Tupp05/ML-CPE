
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):

    # Define class labels
    labels = list(range(len(classes)))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Generate classification report
    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)

    # Generate confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("Confusion Matrix:")
    print(matrix)

    # Save evaluation results
    if save_path:

        output_dir = os.path.dirname(
            os.path.abspath(save_path)
        )

        os.makedirs(output_dir, exist_ok=True)

        # Save confusion matrix
        plot_confusion_matrix(
            matrix,
            classes,
            save_path
        )

        # Save classification report
        report_dict = classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report_dict).transpose()

        report_path = os.path.join(
            output_dir,
            "classification_report.csv"
        )

        report_df.to_csv(report_path, index=True)

        print(f"Saved: {save_path}")
        print(f"Saved: {report_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    # Adjust figure size based on class count
    n_classes = len(classes)
    fig_size = max(6, n_classes * 0.8)

    fig, ax = plt.subplots(
        figsize=(fig_size, fig_size)
    )

    # Display confusion matrix
    image = ax.imshow(
        matrix,
        cmap="Blues"
    )

    # Set class labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Add prediction counts
    threshold = matrix.max() / 2

    for i in range(n_classes):
        for j in range(n_classes):

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                ),
                fontsize=9
            )

    # Add color scale
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path):

    # Get training history
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Create epoch numbers
    epochs = np.arange(1, len(train_acc) + 1)

    # Create accuracy and loss plots
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5)
    )

    # Plot accuracy
    axes[0].plot(
        epochs,
        train_acc,
        label="Training"
    )

    axes[0].plot(
        epochs,
        val_acc,
        label="Validation"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot loss
    axes[1].plot(
        epochs,
        train_loss,
        label="Training"
    )

    axes[1].plot(
        epochs,
        val_loss,
        label="Validation"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Save training history
    os.makedirs(
        os.path.dirname(os.path.abspath(save_path)),
        exist_ok=True
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")
