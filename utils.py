import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_dominant_color(image, k=1):
    """Extract dominant color from an image region (BGR)."""
    image = cv2.resize(image, (20, 20))  # Speed-up
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k).fit(pixels)
    return kmeans.cluster_centers_[0]

