import numpy as np
import logging # Keep logging

logger = logging.getLogger(__name__)

class CustomKMeans: 
    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol  # Tolerance for convergence (squared distance of centroid movement)
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

        if self.random_state is not None:
            np.random.seed(self.random_state)
            # import random # Moved import to top level if not already there
            # random.seed(self.random_state) # For random.sample if used

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initializes centroids by randomly selecting k points from X."""
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            # This case should ideally be caught before calling fit
            logger.error(f"CustomKMeans: Number of samples ({n_samples}) is less than n_clusters ({self.n_clusters}). Cannot initialize centroids properly.")
            # Fallback: if forced, use first k samples, but this is not ideal
            # For robustness, this should raise an error or be handled by the caller.
            # Here, we'll assume the caller checks this.
            # If not, and we must proceed, we could take the first n_samples as centroids if n_samples < n_clusters
            # but that would mean n_clusters effectively becomes n_samples.
            # For now, let's rely on the caller to ensure n_samples >= n_clusters.
            # If not, np.random.choice will fail if replace=False and n_samples < self.n_clusters
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        else:
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[indices]
        return centroids.astype(X.dtype) # Ensure centroids have same dtype as X

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assigns each sample in X to the closest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters), dtype=X.dtype)
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1) # Euclidean distance squared
        labels = np.argmin(distances, axis=1)
        return labels

    def _recalculate_centroids(self, X: np.ndarray, labels: np.ndarray, previous_centroids: np.ndarray) -> np.ndarray:
        """Recalculates centroids as the mean of assigned samples."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]), dtype=X.dtype)
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                logger.warning(f"CustomKMeans: Cluster {i} became empty. Re-using its previous centroid position.")
                new_centroids[i] = previous_centroids[i] # Use the centroid from the previous iteration
        return new_centroids

    def fit(self, X: np.ndarray):
        """Computes K-Means clustering."""
        if X.shape[0] < self.n_clusters:
            # This check is crucial and should ideally be done by the caller,
            # but good to have a safeguard.
            logger.error(f"CustomKMeans fit error: Number of samples ({X.shape[0]}) is less than n_clusters ({self.n_clusters}).")
            # Cannot proceed, perhaps raise ValueError or return self with None attributes
            self.labels_ = np.array([])
            self.cluster_centers_ = np.array([])
            return self # Or raise error

        self.cluster_centers_ = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            old_centroids_for_shift_check = np.copy(self.cluster_centers_)
            self.labels_ = self._assign_clusters(X, self.cluster_centers_)
            self.cluster_centers_ = self._recalculate_centroids(X, self.labels_, old_centroids_for_shift_check) # Pass old for empty handling
            
            centroid_shift = np.sum((self.cluster_centers_ - old_centroids_for_shift_check) ** 2)
            if centroid_shift < self.tol:
                #logger.debug(f"CustomKMeans converged after {iteration+1} iterations. Centroid shift: {centroid_shift:.2e}")
                break
        else: # If loop completes without break (max_iter reached)
            logger.debug(f"CustomKMeans reached max_iter ({self.max_iter}) without converging. Centroid shift: {centroid_shift:.2e}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the closest cluster each sample in X belongs to."""
        if self.cluster_centers_ is None:
            raise ValueError("CustomKMeans model has not been fitted yet. Call fit() before predict().")
        labels = self._assign_clusters(X, self.cluster_centers_)
        return labels
