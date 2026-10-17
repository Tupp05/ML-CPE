
from pathlib import Path
from collections import Counter
import random
import shutil

# Dataset paths
SOURCE_DIR = Path(__file__).resolve().parent / "Dataset_Animal"
OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"

# Class names
CLASSES = ["buffalo", "elephant", "rhino", "zebra"]

# Split configuration
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_data():
    # Collect images and validate YOLO labels
    samples = []

    for class_id, class_name in enumerate(CLASSES):
        class_dir = SOURCE_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Folder not found: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = image_path.with_suffix(".txt")

            if not label_path.exists():
                raise FileNotFoundError(f"Label not found: {label_path}")

            lines = label_path.read_text(encoding="utf-8").splitlines()

            if not lines:
                raise ValueError(f"Empty label: {label_path}")

            # Validate class IDs and bounding boxes
            for line in lines:
                values = line.split()

                if len(values) != 5:
                    raise ValueError(f"Invalid label: {label_path}")

                label_id = int(values[0])
                coords = [float(v) for v in values[1:]]

                if label_id not in range(len(CLASSES)):
                    raise ValueError(f"Invalid class ID: {label_path}")

                x, y, w, h = coords

                if not all(0 <= v <= 1 for v in coords):
                    raise ValueError(f"Invalid coordinates: {label_path}")

                if w <= 0 or h <= 0:
                    raise ValueError(f"Invalid box size: {label_path}")

                if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
                    raise ValueError(f"Box outside image: {label_path}")

                if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                    raise ValueError(f"Box outside image: {label_path}")

            samples.append((image_path, label_path, class_name))

    return samples


def split_data(samples):
    # Split each class into train, validation, and test sets
    rng = random.Random(RANDOM_STATE)
    splits = {"train": [], "val": [], "test": []}

    for class_name in CLASSES:
        class_samples = [
            sample for sample in samples if sample[2] == class_name
        ]

        rng.shuffle(class_samples)

        n = len(class_samples)
        n_train = int(n * TRAIN_SIZE)
        n_val = int(n * VAL_SIZE)

        splits["train"].extend(class_samples[:n_train])
        splits["val"].extend(class_samples[n_train:n_train + n_val])
        splits["test"].extend(class_samples[n_train + n_val:])

    return splits


def save_data(splits):
    # Create folders and copy images with matching labels
    for split_name, samples in splits.items():
        image_dir = OUTPUT_DIR / split_name / "images"
        label_dir = OUTPUT_DIR / split_name / "labels"

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image_path, label_path, class_name in samples:
            # Add class prefix to prevent duplicate filenames
            new_name = f"{class_name}_{image_path.stem}"

            shutil.copy2(
                image_path,
                image_dir / f"{new_name}{image_path.suffix.lower()}"
            )

            shutil.copy2(
                label_path,
                label_dir / f"{new_name}.txt"
            )


def main():
    # Collect and validate dataset
    samples = collect_data()

    print(f"Total images: {len(samples)}")

    # Prevent overwriting existing split folders
    for split_name in ("train", "val", "test"):
        split_dir = OUTPUT_DIR / split_name

        if split_dir.exists():
            raise FileExistsError(
                f"Output folder already exists: {split_dir}"
            )

    # Split and save dataset
    splits = split_data(samples)
    save_data(splits)

    # Display dataset statistics
    for split_name, split_samples in splits.items():
        counts = Counter(sample[2] for sample in split_samples)

        print(f"\n{split_name}: {len(split_samples)} images")

        for class_name in CLASSES:
            print(f"  {class_name}: {counts[class_name]}")

    print("\nDataset splitting completed!")


if __name__ == "__main__":
    main()
