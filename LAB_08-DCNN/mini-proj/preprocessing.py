
import cv2
import numpy as np


def preprocess_image(image, img_size=100):

    # Check invalid images
    if image is None or image.size == 0:
        return None

    # Convert image to RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        return None

    # Resize image
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):

    # Convert images to float32
    images = np.asarray(images, dtype=np.float32)

    # Normalize pixel values to 0-1
    images = images / 255.0

    return np.ascontiguousarray(images)
