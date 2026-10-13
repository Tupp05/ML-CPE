
import os
import cv2
import numpy as np

from preprocessing import preprocess_image

# Define supported image file extensions
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# Read images using NumPy and OpenCV to support Unicode file paths
def read_image(image_path):
    try:
        # Read the image file as binary data using NumPy
        image_data = np.fromfile(
            image_path,
            dtype=np.uint8
        )

        # Decode binary data into an image using OpenCV
        image = cv2.imdecode(
            image_data,
            cv2.IMREAD_COLOR
        )

        return image

    except (OSError, ValueError, cv2.error) as e:
        # Display an error message if the image cannot be read
        print(f"Cannot read image: {image_path}")
        print(f"Reason: {e}")
        return None


# Load images and labels from the dataset directory
def load_data(data_path, img_size=100, max_per_class=None):

    # Check if the dataset directory exists
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}"
        )

    # Create lists to store images and labels
    images = []
    labels = []

    # Detect class names automatically from subfolders
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])

    # Check if the dataset contains class folders
    if not classes:
        raise ValueError("No class folders found.")

    print("Detected classes:", classes)
    print(f"Number of classes: {len(classes)}")

    # Read images from each class folder
    for label, class_name in enumerate(classes):

        # Define the directory path for the current class
        class_path = os.path.join(
            data_path,
            class_name
        )

        # Get image filenames with supported extensions
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        # Count successfully loaded and skipped images
        loaded = 0
        skipped = 0

        for filename in filenames:

            # Stop loading when the class image limit is reached
            if max_per_class is not None and loaded >= max_per_class:
                break

            # Create the full path of the current image
            image_path = os.path.join(
                class_path,
                filename
            )

            # Read the image using read_image() to support Unicode paths
            image = read_image(image_path)

            # Pass the image and image size to preprocess_image()
            # in preprocessing.py
            image = preprocess_image(
                image,
                img_size
            )

            # Skip the image if preprocessing fails
            if image is None:
                skipped += 1
                continue

            # Store the processed image and its class label
            images.append(image)
            labels.append(label)
            loaded += 1

        # Display the number of loaded and skipped images for each class
        print(
            f"Loaded class {class_name}: "
            f"{loaded} images ({skipped} skipped)"
        )

    # Check if any valid images were loaded
    if not images:
        raise ValueError(
            "No valid images found in the dataset."
        )

    # Return images as a NumPy array, labels, and class names
    return (
        np.stack(images),
        np.array(labels),
        classes
    )
