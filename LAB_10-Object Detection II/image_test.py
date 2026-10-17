
from pathlib import Path
import random
import cv2
import pandas as pd
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
IMAGE_DIR = ROOT / "dataset" / "test" / "images"
COMPARISON_CSV = RUNS_DIR / "comparison_results.csv"

# Test configuration
N_SAMPLES = 5
IMG_SIZE = 640
CONFIDENCE = 0.25
RANDOM_STATE = 42


def get_best_model():
    # Load comparison results
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(f"File not found: {COMPARISON_CSV}")

    df = pd.read_csv(COMPARISON_CSV)
    df.columns = df.columns.str.strip()

    # Select model with highest mAP50-95
    best_row = df.loc[df["mAP50-95"].idxmax()]
    model_name = str(best_row["Model"])

    # Get trained model path
    weights = RUNS_DIR / model_name / "weights" / "best.pt"

    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")

    print(f"Best Model: {model_name}")
    print(f"Test mAP50-95: {best_row['mAP50-95']:.4f}")

    return weights


def main():
    # Select best trained model
    weights = get_best_model()

    # Check test dataset
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {IMAGE_DIR}")

    # Load trained model
    model = YOLO(str(weights))

    # Get test images
    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        raise RuntimeError("No test images found")

    # Randomly select test images
    random.seed(RANDOM_STATE)
    selected_images = random.sample(
        images, min(N_SAMPLES, len(images))
    )

    # Object detection
    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            device=0,
            save=True,
            project=str(RUNS_DIR),
            name="image_test",
            exist_ok=True
        )

        for result in results:
            print(f"\nImage: {image.name}")

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                print(
                    f"{model.names[class_id]}: "
                    f"{confidence:.2f}"
                )

            # Display detection result
            annotated = result.plot()
            cv2.imshow("YOLO Image Detection", annotated)

            # Wait for key or window close
            while True:
                key = cv2.waitKey(100) & 0xFF

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return

                if key != 255:
                    break

                if cv2.getWindowProperty(
                    "YOLO Image Detection", cv2.WND_PROP_VISIBLE
                ) < 1:
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()
    print("\nImage testing completed!")


if __name__ == "__main__":
    main()
