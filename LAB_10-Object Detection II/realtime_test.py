
from pathlib import Path
import csv
import time
import cv2
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
COMPARISON_CSV = RUNS_DIR / "comparison_results.csv"

# Detection configuration
CAMERA_INDEX = 0
IMG_SIZE = 640
CONFIDENCE = 0.25
DEVICE = 0
WINDOW_NAME = "YOLOv8 Real-time Animal Detection"


def get_best_model():
    # Load comparison results
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"Comparison file not found: {COMPARISON_CSV}"
        )

    with open(COMPARISON_CSV, "r", encoding="utf-8-sig", newline="") as file:
        results = list(csv.DictReader(file))

    if not results:
        raise RuntimeError("No comparison results found")

    # Select model with highest mAP50-95
    best = max(
        results,
        key=lambda row: float(row["mAP50-95"])
    )

    model_name = best["Model"]
    weights = RUNS_DIR / model_name / "weights" / "best.pt"

    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")

    print(f"Best Model: {model_name}")
    print(f"mAP50-95: {float(best['mAP50-95']):.4f}")

    return weights


def main():
    # Load best trained model
    weights = get_best_model()
    model = YOLO(str(weights))

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open webcam (camera index {CAMERA_INDEX})"
        )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Read frame
            ok, frame = cap.read()

            if not ok:
                print("Cannot read camera frame")
                break

            # Object detection
            start_time = time.perf_counter()

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False
            )

            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Display FPS
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display result
            cv2.imshow(WINDOW_NAME, annotated_frame)

            # Press Q to exit
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # Exit when window is closed
            if cv2.getWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        # Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()

    print("\nReal-time testing completed!")


if __name__ == "__main__":
    main()
