
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
OUTPUT_DIR = RUNS_DIR / "comparison_plots"

# Experiment configuration
EPOCHS_LIST = [10, 20, 30]

# Metrics to compare
METRICS = {
    "train/box_loss": "Training Box Loss",
    "train/cls_loss": "Training Classification Loss",
    "train/dfl_loss": "Training DFL Loss",
    "val/box_loss": "Validation Box Loss",
    "val/cls_loss": "Validation Classification Loss",
    "val/dfl_loss": "Validation DFL Loss",
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95"
}


def load_results():
    # Load training results
    results = {}

    for epochs in EPOCHS_LIST:
        csv_path = RUNS_DIR / f"animal_epochs{epochs}" / "results.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        results[epochs] = df

    return results


def plot_metrics(results):
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for column, title in METRICS.items():
        plt.figure(figsize=(10, 6))

        # Plot each experiment
        for epochs, df in results.items():
            if column not in df.columns:
                raise KeyError(f"Missing column: {column}")

            plt.plot(
                df["epoch"],
                df[column],
                label=f"{epochs} Epochs",
                linewidth=2
            )

        # Configure plot
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close()

        print(f"Saved: {filename}")


def main():
    # Load experiment results
    results = load_results()

    # Generate comparison plots
    plot_metrics(results)

    print("\nAll comparison plots completed!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
