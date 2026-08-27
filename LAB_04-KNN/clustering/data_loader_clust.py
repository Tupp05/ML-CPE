import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Input the main folder paths containing dog images
FOLDER_PATHS = [r"data_dog-breeds"]

IMG_SIZE = (64, 64)
CLASS_NAMES = ["beagle", "bulldog", "dalmatian", "german-shepherd", "husky", "labrador-retriever", "poodle", "rottweiler"]

# ---------------------------------------------------------------------------
# Functions for loading and preparing data
def load_data():
    """
    Returns a dict containing:
        X        : Scaled images (0-1) (Used as input for KMeans clustering model)
        X_raw    : Original pixel values (0-255) (For plotting/displaying images)
        df       : DataFrame containing file paths and class names (For verification)
    """
    X_raw_list = []
    paths = []
    labels = []
    
    # Loops to read data from every folder defined in FOLDER_PATHS
    for folder in FOLDER_PATHS:
        base_dir = Path(folder)
        
        # Notification when the folder is worng or not found the folder
        if not base_dir.exists():
            print(f"Warning: Could not find folder '{folder}'. Skip to the next folder.")
            continue
            
        for class_name in CLASS_NAMES:
            class_dir = base_dir / class_name
            if not class_dir.exists():
                continue 
                
            # Retrieve all image files in that class (.jpg, .png, .jpeg)
            for img_path in class_dir.glob("*.*"):
                # Read file via numpy to handle non-English/Thai file paths
                img_array = np.fromfile(str(img_path), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Check read file is successfully
                if img is not None:
                    # Resize the image
                    img_resized = cv2.resize(img, IMG_SIZE)
                    # Convert 2D Images to 1D Array (Flatten)
                    X_raw_list.append(img_resized.flatten())
                    
                    # Store reference data
                    paths.append(str(img_path.name)) # Keep only the file name
                    labels.append(class_name)

    # Convert List to Numpy Array
    X_raw = np.array(X_raw_list, dtype="float32")
    
    # Scaling (Normalize the image to be in the range 0 - 1)
    X = X_raw / 255.0

    # Create a DataFrame for later verification
    df = pd.DataFrame({
        "filename": paths,
        "true_label": labels
    })

    return {"X": X, "X_raw": X_raw, "df": df}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading images for clustering...")
    data = load_data()
    
    print("\nData loaded successfully!")
    print("Data shape (Number of images, Number of features):", data["X"].shape)
    
    print(f"Minimum value of X: {data['X'].min()} | Maximum value of X: {data['X'].max()}")
    
    print("\nSample data in DataFrame (df):")
    print(data["df"].head())
    print("\nNumber of images loaded for each class:")
    print(data["df"]["true_label"].value_counts())