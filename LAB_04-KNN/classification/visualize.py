import matplotlib
matplotlib.use("Agg")    

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
def plot_elbow(k_values, inertias, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, inertias, "o-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ---------------------------------------------------------------------------
def plot_cluster_centroids(centroids, out_path, img_size=(64, 64)):
    """
    New function for images: Reconstruct the Centroid values back into images 
    to see what the average dog face looks like to the model in each cluster.
    """
    n_clusters = len(centroids)
    # Arrange images into rows and columns
    cols = 4
    rows = (n_clusters + cols - 1) // cols
    
    plt.figure(figsize=(3 * cols, 3 * rows))
    
    for i, centroid in enumerate(centroids):
        plt.subplot(rows, cols, i + 1)
        
        # Reshape 1D array (12288 elements) back to 3D (width 64 x height 64 x 3 colors)
        img = centroid.reshape(img_size[0], img_size[1], 3)
        
        # Convert color from BGR (OpenCV standard) to RGB (for Matplotlib to display correctly)
        img = img[:, :, ::-1] 
        
        # Show the image (use np.clip to prevent values from exceeding the 0-1 range)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Cluster {i} Average")
        plt.axis("off") # Turn off X and Y axis scales since these are images
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()