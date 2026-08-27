import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

FOLDER_PATHS = [r"data_dog-breeds"]
IMG_SIZE = (128, 128)
CLASS_NAMES = ["beagle", "bulldog", "dalmatian", "german-shepherd", "husky", "labrador-retriever", "poodle", "rottweiler"]


# Functions for loading and preparing dataFunctions for loading and preparing data

def load_and_preprocess_images(test_size=0.2, seed=42):
    X = []
    y = []
    
    # Loops to read data from every folder defined in FOLDER_PATHS
    for folder in FOLDER_PATHS:
        base_dir = Path(folder)
        
        # Notification when the folder is worng or not found the folder
        if not base_dir.exists():
            print(f"Warning: Could not find folder '{folder}'. Skip to the next folder.")
            continue
            
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = base_dir / class_name
            if not class_dir.exists():
                continue 
                
            # Retrieve all image files in that class (.jpg, .png, .jpeg)
            for img_path in class_dir.glob("*.*"):
                img = cv2.imread(str(img_path))
                
                # Check read file is successfully
                if img is not None:
                    # Resize the image
                    img_resized = cv2.resize(img, IMG_SIZE)
                    # Convert 2D Images to 1D Array (Flatten) for KNN
                    X.append(img_resized.flatten())
                    y.append(label)

    # Convert List to Numpy Array for use with Sklearn
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")

    # If there is no information at all Stop working
    if len(X) == 0:
        raise ValueError("No images found. Please check FOLDER_PATHS and CLASS_NAMES again")

    # Data split: X_temp, y_temp (80%) and X_test, y_test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    
    # split X_temp (80%) into Train (60%) and Val (20%).
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp)

    # Scaling (Normalize the image to be in the range 0 - 1)
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    return X_train, X_val, X_test, y_train, y_val, y_test
