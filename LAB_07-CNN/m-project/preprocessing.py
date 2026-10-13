
import cv2
import numpy as np


# Preprocess images by converting color formats and resizing
def preprocess_image(image, img_size=100):
    """Resize image to RGB format."""

    # Check if the image is empty or cannot be loaded
    if image is None or image.size == 0:
        return None

    # Convert grayscale images to RGB
    if image.ndim == 2:
        image = cv2.cvtColor(
            image, cv2.COLOR_GRAY2RGB
        )

    # Convert images with an Alpha Channel (BGRA) to RGB
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(
            image, cv2.COLOR_BGRA2RGB
        )

    # Convert normal color images from BGR to RGB
    else:
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

    # Get the original height and width of the image
    height, width = image.shape[:2]

    # Select interpolation based on the original image size
    # Use INTER_AREA for downscaling and INTER_LINEAR for upscaling
    interpolation = (
        cv2.INTER_AREA
        if height > img_size or width > img_size
        else cv2.INTER_LINEAR
    )

    # Resize the image to the specified width and height
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=interpolation
    )

    return image


# Convert images into a contiguous NumPy array for CNN input
def to_features(images):

    # Keep pixel values as uint8 (0-255) to reduce memory usage
    # Normalization from 0-255 to 0-1 is performed inside the CNN model
    return np.ascontiguousarray(
        images, dtype=np.uint8
    )
