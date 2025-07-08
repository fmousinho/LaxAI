import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP 
import cv2

logger = logging.getLogger(__name__)

_KERNEL_OPEN = np.ones((7, 7), np.uint8)  # Kernel for morphological operations

class PlayerMasker:
    """    
    A class to handle player masking in video frames using KMeans clustering and morphological operations.
    This class is designed to process crops of player images, apply KMeans clustering to distinguish between players and grass, and generate masks for each player.
    """

    def __init__(self):
        self.predictor = None
        self.type = "kmeans_morph_edge"

    def fit_predict(self, *args, **kwargs) -> List[np.ndarray]:
        """
        Placeholder for the fit_predict method.
        This method should be overridden by subclasses to implement specific masking logic.
        """
        if self.type == "kmeans_morph_edge":
            if len(args) < 1 or not isinstance(args[0], list):
                raise ValueError("Expected a list of crops as the first argument.")
            if not all(isinstance(crop, np.ndarray) for crop in args[0]):
                raise ValueError("All elements in the crops list must be numpy arrays.")
            return self._fit_predict_kmeans_morph_edge(*args, **kwargs)
        else:
            logger.error(f"Unknown type '{self.type}' for PlayerMasker. Please implement the fit_predict method for this type.")
            raise NotImplementedError("fit_predict method must be implemented in subclasses.")

    def _fit_predict_kmeans_morph_edge(
        self,
        crops: List[np.ndarray]
        ) -> List[np.ndarray]:
        """
        Trains the player masker using the provided crops and method type.

        Args:
            crops (List[np.ndarray]): The input image crops.
            type (str): The method type for training.
            video (Optional[str]): The video identifier (if applicable).

        Returns:
            Dict[str, np.ndarray]: The trained mask for each crop.
        """
        shapes = []
        roi_crops = crops.copy()
        flat_crop_array = []
        for crop in roi_crops:
            shapes.append(crop.shape)
            flat_crop = crop.reshape(-1, 3).astype(np.float32) / 255.0
            flat_crop_array.append(flat_crop)
        
        flat_crop_norm = np.concatenate(flat_crop_array, axis=0)
        self.predictor = KMeans(n_clusters=2, random_state=42)
        player_or_grass_mask = self.predictor.fit_predict(flat_crop_norm)

        masked_crops = []
        start = 0
        kernel_open = _KERNEL_OPEN
        
        for i, shape in enumerate(shapes):
            stop = start + shape[0] * shape[1]
            flat_mask = player_or_grass_mask[start:stop]
            mask = np.reshape(flat_mask, shape[:2])
            mask_edges = set([mask[0, 0], mask[0, -1], mask[-1, 0], mask[-1, -1]])
            grass_cluster = max(mask_edges, key=list(mask_edges).count)
            player_img_cluster = 1 - grass_cluster
            mask = (mask == player_img_cluster).astype(np.uint8)

            # Clean the mask using morphological opening to remove noise.
            # The mask should have player pixels as 255 and background as 0.
            cleaned_mask = cv2.morphologyEx(mask * 255, cv2.MORPH_OPEN, kernel_open)
            
            # Apply the cleaned mask to the original crop.
            masked_crop = roi_crops[i]
            masked_crop = cv2.bitwise_and(masked_crop, masked_crop, mask=cleaned_mask)
            masked_crop[cleaned_mask == 0] = [255, 255, 255]  # Set background to white.
            masked_crops.append(masked_crop.copy())
            start = stop

        return masked_crops
    





class TeamIdentifier:
    """
    Manages team identification for players using clustering on their embeddings.
    """

    def __init__(self, n_clusters: int = 2): 
        """
        Initializes the TeamIdentifier module.

        Args:
            n_clusters (int): The number of teams to identify. Defaults to 2.
            umap_n_components (int): The number of dimensions for UMAP reduction.
                                     Should be >= n_clusters.
        """

        self.n_clusters = n_clusters
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42)


    def fit_predict (self, embeddings_array: np.ndarray) -> np.ndarray:
        """
        Discovers teams by clustering player embeddings using UMAP for dimensionality
        reduction and KMeans for clustering. Enables the predict functions.

        Args:
            embeddings_array (np.ndarray): A 2D array where each row is an embedding.

        Returns:
            np.ndarray: A 2D array of team IDs (0 or 1) for each input embedding
    
        """

        if embeddings_array is None or len(embeddings_array) == 0:
            return np.array([])

        teams = self.kmeans_model.fit_predict(embeddings_array)
        return teams
    

