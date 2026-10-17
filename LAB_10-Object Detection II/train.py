
from ultralytics import YOLO
from pathlib import Path

# Training configuration
DATA_PATH = Path(__file__).resolve().parent / "dataset" / "data.yaml"
MODEL_NAME = "yolov8n.pt"

EPOCHS_LIST = [10, 20, 30]
IMG_SIZE = 640
BATCH_SIZE = 8
RANDOM_STATE = 42


def main():
    # Train models with different epochs
    for epochs in EPOCHS_LIST:
        print(f"\nTraining YOLOv8 with {epochs} epochs")

        # Load a fresh pretrained model
        model = YOLO(MODEL_NAME)

        # Train model
        model.train(
            data=str(DATA_PATH),
            epochs=epochs,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=0,
            workers=0,
            amp=True,
            cache=False,
            seed=RANDOM_STATE,
            deterministic=True,
            project=str(Path(__file__).resolve().parent / "runs"),
            name=f"animal_epochs{epochs}",
            exist_ok=False,
            save=True,
            plots=True
        )

        # Get output directory
        save_dir = Path(model.trainer.save_dir)

        print(f"\nTraining completed: {epochs} epochs")
        print(f"Best Model: {save_dir / 'weights/best.pt'}")
        print(f"Last Model: {save_dir / 'weights/last.pt'}")
        print(f"Training CSV: {save_dir / 'results.csv'}")

    print("\nAll training experiments completed!")


if __name__ == "__main__":
    main()
