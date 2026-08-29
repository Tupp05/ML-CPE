import os
import cv2
import csv
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path, img_size=100, max_per_class=None):
    images = []
    labels = []

    # Find all Folder IDs (e.g., 18, 20)
    folder_ids = set()
    for split_folder in ['train', 'test']:
        split_path = os.path.join(data_path, split_folder)
        if os.path.exists(split_path):
            folders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
            folder_ids.update(folders)
            
    folder_ids = sorted(list(folder_ids))

    # Read the CSV file to create a mapping (Folder ID -> Species Name)
    class_mapping = {}
    # Point to the train.csv file in the csv folder
    csv_path = os.path.join(data_path, 'csv', 'train.csv') 
    
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Extract the header row
            try:
                # Find the column indices for 'class_id' and 'binomial'
                idx_class_id = header.index('class_id')
                idx_binomial = header.index('binomial')
                for row in reader:
                    # Store in a dictionary, e.g., { '18': 'Agkistrodon contortrix' }
                    class_mapping[str(row[idx_class_id])] = row[idx_binomial]
            except ValueError:
                print("Warning: Could not find 'class_id' or 'binomial' in CSV.")

    # Create a list of human-readable class names
    classes = []
    for fid in folder_ids:
        # Use the species name if found in the CSV; otherwise, default to 'Class' + ID
        name = class_mapping.get(fid, f"Class {fid}")
        classes.append(name)
        
    print("Detected species:", classes)

    # Load images from each class directory
    for label, folder_id in enumerate(folder_ids):
        species_name = classes[label]
        loaded = 0
        skipped = 0
        
        for split_folder in ['train', 'test']:
            # Use the numeric folder ID to access the directory path
            class_path = os.path.join(data_path, split_folder, folder_id) 
            
            if not os.path.exists(class_path):
                continue
                
            filenames = sorted(
                f for f in os.listdir(class_path)
                if f.lower().endswith(VALID_EXT)
            )

            for filename in filenames:
                if max_per_class and loaded >= max_per_class:
                    break

                image_path = os.path.join(class_path, filename)
                # Read raw data with np.fromfile first, then decode (supports non-ASCII paths)
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                image = preprocess_image(image, img_size)

                if image is None:
                    skipped += 1
                    continue

                images.append(image)
                # Neural Networks require integer labels (0, 1, 2...) based on the index
                labels.append(label) 
                loaded += 1

        print(f"Loaded {species_name} (Folder {folder_id}): {loaded} images ({skipped} skipped)")

    if len(images) == 0:
        raise ValueError(f"No images found in {data_path}. Please check your dataset path.")

    return np.stack(images), np.array(labels), classes