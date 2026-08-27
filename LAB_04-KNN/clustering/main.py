# STEP 1  Load the dataset.
# STEP 2  Find the best k using the Elbow Method.
# STEP 3  Run K-Means with the selected k.
# STEP 4  Analyze each cluster.
# STEP 5  Use KNN to classify a new image.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import data_loader_clust as data_loader
import visualize
from kmeans_tf import TFKMeans
from knn_tools import KNNClusterAssigner

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Set K to 8 initially because we know there are 8 dog breeds
N_CLUSTERS = 8     
KNN_K = 5          


def title(text):
    print("\n" + "--" * 30)
    print(text)

# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1: Load data")
    data = data_loader.load_data()
    X = data["X"]                # Scaled data (used for calculation)
    X_raw = data["X_raw"]        # Original pixel values 
    df = data["df"]              # DataFrame storing filenames and breed names

    print(f"Data size: {X.shape[0]} images x {X.shape[1]} pixels (features)")


    # =====================================================================
    title("STEP 2: How many clusters should we split into?")
    # =====================================================================
    k_values = [2, 4, 6, 8, 10, 12, 15] # Adjust k values to cover the number of breeds (8)
    inertias = []

    for k in k_values:
        km = TFKMeans(n_clusters=k).fit(X)
        sil = silhouette_score(X, km.labels_)
        inertias.append(km.inertia_)
        print(f"   k = {k:>2}  ->  inertia = {km.inertia_:8.1f}   silhouette = {sil:.3f}")

    visualize.plot_elbow(k_values, inertias, OUT_DIR / "01_elbow.png")
    print(f"\nSaved graph to outputs/01_elbow.png")
    print(f"(Choosing k = {N_CLUSTERS} for the next clustering step)")


    # =====================================================================
    title(f"STEP 3: Run K-Means (k = {N_CLUSTERS})")
    # =====================================================================
    km = TFKMeans(n_clusters=N_CLUSTERS)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"Used {km.n_iter_} iterations until centroids stabilized")
    print(f"Inertia          : {km.inertia_:.1f}")
    print(f"Silhouette score : {sil:.3f}")
    print(f"Number of members in each cluster: {np.bincount(labels).tolist()}")

    if sil < 0.15: # Images often get low Silhouette scores because the data is highly complex
        print("\n[Note] Raw Pixels have very high complexity.")
        print("       Therefore, the Silhouette score is usually lower than standard numerical data.")

    # visualize.plot_clusters(X_raw[:, [1, 2]], labels, OUT_DIR / "02_clusters.png")
    # print(" (Skipping 2D Scatter plot because image data has too many dimensions)")

    visualize.plot_cluster_centroids(km.centroids_, OUT_DIR / "02_cluster_centroids.png")
    print("Saved the average dog face of each cluster to '02_cluster_centroids.png'")


    # =====================================================================
    title("STEP 4: What are the characteristics of each group?")
    # =====================================================================
    # Instead of finding numerical averages, let's see which breeds ended up in each group
    df["cluster"] = labels
    
    # Create a frequency distribution table (Crosstab) between the assigned Cluster and true breed
    summary = pd.crosstab(df["cluster"], df["true_label"])
    
    print("\nTable showing the number of images of each breed assigned to each group:")
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "cluster_distribution.csv")
    print("\nSaved distribution table to 'cluster_distribution.csv'")


    # =====================================================================
    title(f"STEP 5: Use KNN to assign new images into groups (k = {KNN_K})")
    # =====================================================================
    n_known = int(len(X) * 0.8)
    X_known, labels_known = X[:n_known], labels[:n_known]
    X_new, labels_new = X[n_known:], labels[n_known:]

    assigner = KNNClusterAssigner(k=KNN_K)
    assigner.fit(X_known, labels_known)
    knn_pred = assigner.predict(X_new)

    accuracy = float(np.mean(knn_pred == labels_new))
    print(f"Number of 'new images': {len(X_new)} images")
    print(f"KNN assigned groups correctly compared to K-Means: {accuracy * 100:.1f} %")
    print("Use KNN for new data without rerunning K-Means.")

    # =====================================================================
    title("Save results to CSV file")
    # =====================================================================
    result = df.copy()
    result.to_csv(OUT_DIR / "clustered_images.csv",
                  index=False, encoding="utf-8-sig")

    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()