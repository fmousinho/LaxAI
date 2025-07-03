import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP # Make sure to install umap-learn: pip install umap-learn

logger = logging.getLogger(__name__)

class TeamIdentification:
    """
    Manages team identification for players using clustering on their embeddings.
    """

    def __init__(self, n_clusters: int = 2, umap_n_components: int = 3): 
        """
        Initializes the TeamIdentification module.

        Args:
            n_clusters (int): The number of teams to identify. Defaults to 2.
            umap_n_components (int): The number of dimensions for UMAP reduction.
                                     Should be >= n_clusters.
        """
        if umap_n_components < n_clusters:
            logger.warning(f"UMAP n_components ({umap_n_components}) is less than n_clusters ({n_clusters}). "
                           f"This might lead to suboptimal clustering. Setting umap_n_components to n_clusters.")
            self.umap_n_components = n_clusters
        else:
            self.umap_n_components = umap_n_components

        self.n_clusters = n_clusters
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.umap_model = UMAP(n_components=self.umap_n_components, random_state=42)


    def train_team_identification (self, embeddings_array: np.ndarray):
        """
        Discovers teams by clustering player embeddings using UMAP for dimensionality
        reduction and KMeans for clustering. Enables the predict functions.

        Args:
            embeddings_array (np.ndarray): A 2D array where each row is an embedding.
    
        """

        if embeddings_array is None or len(embeddings_array) == 0:
            return

         # --- Apply UMAP for dimensionality reduction --

        logger.info(f"Applying UMAP to reduce embeddings from {embeddings_array.shape[1]} to {self.umap_n_components} dimensions.")

        try:
            reduced_embeddings = self.umap_model.fit_transform(embeddings_array)
        except Exception as e:
            logger.error(f"Error during UMAP transformation: {e}.")
            self.umap_model = None # Reset model if fit fails
            return
 
        # --- Apply KMeans clustering ---
        logger.info(f"Applying KMeans with {self.n_clusters} clusters.")
   
        try:
            team_labels = self.kmeans_model.fit_predict(reduced_embeddings)
        except Exception as e:
            logger.error(f"Error during KMeans clustering: {e}.")
            self.kmeans_model = None # Reset model if fit fails
            return



    def predict_team(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predicts the team for one or more player embeddings.

        Args:
            embeddings (np.ndarray): 2D array where each row is an embedding.

        Returns:
            np.ndarray: An array of predicted team IDs (0 or 1) for each input embedding -1 is returned if there is an issue.
        """
        if self.umap_model is None or self.kmeans_model is None:
            logger.warning("UMAP or KMeans models are not trained. Call discover_teams first.")
            return np.full((embeddings.shape[0],), -1, dtype=int)

        try:
            # Ensure embeddings is 2D (batch_size, embedding_dim)
            if embeddings.ndim == 1:
                embeddings_input = embeddings.reshape(1, -1)
            elif embeddings.ndim == 2:
                embeddings_input = embeddings
            else:
                logger.error(f"Invalid embedding dimensions: {embeddings.ndim}. Expected 1 or 2.")
                return None

            reduced_embeddings = self.umap_model.transform(embeddings_input)
            predicted_teams = self.kmeans_model.predict(reduced_embeddings)
            return predicted_teams.astype(int)

        except Exception as e:
            logger.error(f"Error predicting team for embeddings: {e}")
            return None