
from pathlib import Path
import csv
from ultralytics import YOLO

# Evaluation configuration
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "data.yaml"
RUNS_DIR = ROOT / "runs"

EPOCHS_LIST = [10, 20, 30]
IMG_SIZE = 640
BATCH_SIZE = 8


def main():
    # Store evaluation results
    results = []

    # Evaluate each trained model
    for epochs in EPOCHS_LIST:
        model_name = f"animal_epochs{epochs}"
        weights = RUNS_DIR / model_name / "weights" / "best.pt"

        if not weights.exists():
            raise FileNotFoundError(f"Model not found: {weights}")

        # Load trained model
        model = YOLO(str(weights))

        # Evaluate on test dataset
        metrics = model.val(
            data=str(DATA_PATH),
            split="test",
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=0,
            workers=0,
            plots=True,
            project=str(RUNS_DIR),
            name=f"evaluation_epochs{epochs}",
            exist_ok=True
        )

        # Collect evaluation metrics
        result = {
            "Model": model_name,
            "Epochs": epochs,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr,
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map
        }

        results.append(result)

        # Display evaluation results
        print(f"\n===== {model_name} =====")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall: {result['Recall']:.4f}")
        print(f"mAP50: {result['mAP50']:.4f}")
        print(f"mAP50-95: {result['mAP50-95']:.4f}")

    # Save comparison results
    csv_path = RUNS_DIR / "comparison_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Select best model by mAP50-95
    best_model = max(results, key=lambda x: x["mAP50-95"])

    print("\n===== Best Model =====")
    print(f"Model: {best_model['Model']}")
    print(f"mAP50-95: {best_model['mAP50-95']:.4f}")
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
