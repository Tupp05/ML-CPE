import matplotlib
matplotlib.use("Agg")    

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from data_loader_class import load_and_preprocess_images, CLASS_NAMES

# function Evaluate

def plot_k_curve(k_values, scores, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, scores, "o-")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Validation accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 7)) 
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    
    # Adjust the X-axis angle to 45 degree
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True label")

    # Add numbers in the box.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            # Adjust the font color to contrast with the background.
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return cm

def print_report(y_true, y_pred, class_names):
    print("\n" + "="*50)
    print("Classification Report (Test Set)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

def save_predictions(y_true, y_pred, class_names, out_path):
    df = pd.DataFrame({
        "true_label": [class_names[i] for i in y_true],
        "predicted_label": [class_names[i] for i in y_pred],
        "correct": y_true == y_pred,
    })
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df

# Train (Main Execution)

if __name__ == "__main__":
    print("Loading image data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_images()
    
    print("\nSearching for the best K value with Validation Set...")
    k_values = [1, 3, 5, 7, 9, 11, 15]
    val_scores = []
    
    best_k = 1
    best_score = 0
    
    # Train with difference K value
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Validation Set
        score = knn.score(X_val, y_val)
        val_scores.append(score)
        print(f"   -> k={k:2d} Accuracy score: {score*100:.2f}%")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\n The best K value is K={best_k} (Accuracy {best_score*100:.2f}%)")
    plot_k_curve(k_values, val_scores, "k_curve.png")
    print("Save K-curve to 'k_curve.png'")
    
    # ---------------------------------------------------------
    print("\nTrain the models with the best K and measure results with Test Sets....")
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)
    
    # Prediction
    y_pred = final_knn.predict(X_test)
    
    # Call the evaluation function
    print_report(y_test, y_pred, CLASS_NAMES)
    
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, "confusion_matrix.png")
    print("Save the Confusion Matrix as 'confusion_matrix.png'")
    
    save_predictions(y_test, y_pred, CLASS_NAMES, "predictions.csv")
    print("Save the prediction results for each image 'predictions.csv'")