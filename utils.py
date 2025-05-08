import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D scatter plot
import logging

logger = logging.getLogger(__name__)

def plot_team_kmeans_clusters(colors_array: np.ndarray, labels: np.ndarray, centers: np.ndarray, output_filename: str = "team_kmeans_visualization.png"):
    """
    Generates and saves a 3D scatter plot of color clusters.

    Args:
        colors_array (np.ndarray): Array of RGB colors (N, 3).
        labels (np.ndarray): Cluster labels for each color.
        centers (np.ndarray): Cluster centers (n_clusters, 3).
        output_filename (str): Filename to save the plot.
    """
    if not (colors_array.ndim == 2 and colors_array.shape[1] == 3):
        logger.warning("Cannot plot KMeans clusters: colors_array is not 3D (RGB).")
        return
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_colors = ['blue', 'red'] 
        for i in range(len(colors_array)):
            ax.scatter(colors_array[i, 0], colors_array[i, 1], colors_array[i, 2], 
                       color=plot_colors[labels[i]], marker='o', alpha=0.6)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   marker='x', s=200, color='black', label='Team Color Centers')
        ax.set_xlabel('Red Channel')
        ax.set_ylabel('Green Channel')
        ax.set_zlabel('Blue Channel')
        ax.set_title('KMeans Clustering of Player Dominant Colors for Teams')
        plt.legend()
        plt.savefig(output_filename)
        plt.close(fig) 
        logger.info(f"KMeans team clustering visualization saved to {output_filename}")
    except Exception as e:
        logger.error(f"Error generating KMeans cluster plot: {e}", exc_info=True)
