
import os
import cv2
import numpy as np

from preprocessing import preprocess_image


# Supported image formats
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_data(data_path, img_size=100, max_per_class=None):

    # Check dataset directory
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}"
        )

    images = []
    labels = []

    # Detect Classes
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])

    if not classes:
        raise ValueError("No class folders found in dataset.")

    print("Detected classes:", classes)

    # Load Images
    for label, class_name in enumerate(classes):

        class_path = os.path.join(data_path, class_name)

        filenames = sorted([
            filename
            for filename in os.listdir(class_path)
            if filename.lower().endswith(VALID_EXT)
        ])

        loaded = 0
        skipped = 0

        for filename in filenames:

            if max_per_class is not None and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)

            # Read image with Unicode path support
            try:
                image_data = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            except (OSError, cv2.error, ValueError):
                image = None

            # Skip unreadable images
            if image is None:
                skipped += 1
                continue

            # Preprocess Image
            image = preprocess_image(image, img_size)

            # Skip invalid images
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(
            f"Loaded class {class_name}: "
            f"{loaded} images ({skipped} skipped)"
        )

    # Convert to NumPy Arrays
    if not images:
        raise ValueError("No valid images found in dataset.")

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)

    print(f"\nTotal images loaded: {len(images)}")
    print(f"Image shape: {images.shape}")
    print(f"Total classes: {len(classes)}")

    return images, labels, classes
