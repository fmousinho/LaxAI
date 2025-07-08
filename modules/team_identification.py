import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP 
import cv2

logger = logging.getLogger(__name__)

_KERNEL_OPEN = np.ones((5, 5), np.uint8)  # Kernel for morphological operations

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
        elif self.type == "grass_avg":
            if len(args) < 1 or not isinstance(args[0], list):
                raise ValueError("Expected a list of crops as the first argument.")
            if not all(isinstance(crop, np.ndarray) for crop in args[0]):
                raise ValueError("All elements in the crops list must be numpy arrays.")
            # Check for optional frames argument
            frames = kwargs.get('frames', None)
            if frames is None or len(frames) == 0:
                logger.warning("No frames provided for grass_avg method, falling back to edge pixels from crops")
            return self._fit_predict_grass_avg(*args, **kwargs)
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
        crops = crops.copy()
        # Convert to LAB color space for better clustering
        roi_crops_lab = [cv2.cvtColor(crop, cv2.COLOR_BGR2LAB) for crop in crops]

        flat_crop_array = []
        for crop in roi_crops_lab:
            shapes.append(crop.shape)
            # Normalize LAB values (OpenCV uses 0-255 for all LAB channels)
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
            
            # Apply the cleaned mask to the original BGR crop (not LAB)
            masked_crop = crops[i]  # Use original BGR crop
            masked_crop = cv2.bitwise_and(masked_crop, masked_crop, mask=cleaned_mask)
            masked_crop[cleaned_mask == 0] = [255, 255, 255]  # Set background to white in BGR
            masked_crop = masked_crop.copy()
            masked_crops.append(masked_crop)
            start = stop

        return masked_crops
    
    def _fit_predict_grass_avg(
        self,
        crops: List[np.ndarray],
        frames: Optional[List[np.ndarray]] = None
        ) -> List[np.ndarray]:
        """
        Creates player masks by identifying grass regions using average color statistics.
        
        This method calculates the average grass color and standard deviation from the bottom
        half of 5 different frames (beginning, middle, end of video segment), then creates 
        masks by identifying pixels that fall within 2 standard deviations of the grass color.

        Args:
            crops (List[np.ndarray]): The input image crops.
            frames (Optional[List[np.ndarray]]): List of frames to sample grass color from.
                                               If None, falls back to edge pixels from crops.

        Returns:
            List[np.ndarray]: The masked crops with player regions isolated.
        """
        crops = crops.copy()
        masked_crops = []
        
        # Collect grass pixels from frame samples or fallback to crop edges
        if frames is not None and len(frames) > 0:
            # Use bottom half of 5 frames sampled from the video segment
            num_frames = len(frames)
            if num_frames >= 5:
                # Sample 5 frames: beginning, quarter, middle, three-quarter, end
                indices = [
                    0,                              # Beginning
                    num_frames // 4,                # Quarter
                    num_frames // 2,                # Middle  
                    3 * num_frames // 4,            # Three-quarter
                    num_frames - 1                  # End
                ]
            else:
                # Use all available frames if less than 5
                indices = list(range(num_frames))
            
            all_grass_pixels = []
            for idx in indices:
                frame = frames[idx]
                # Extract bottom half of the frame (assuming it contains mostly grass)
                h, w = frame.shape[:2]
                bottom_half = frame[h//2:, :]  # Bottom 50% of frame
                
                # Reshape to get all pixels in bottom half
                grass_pixels = bottom_half.reshape(-1, 3)
                all_grass_pixels.append(grass_pixels)
            
            # Combine all grass pixels from sampled frames
            all_grass_pixels = np.concatenate(all_grass_pixels, axis=0).astype(np.float32)
            
        else:
            # Fallback: collect edge pixels from all crops to estimate grass color
            logger.warning("No frames provided, using edge pixels from crops for grass estimation")
            all_edge_pixels = []
            for crop in crops:
                # Extract edge pixels (assuming they're mostly grass)
                h, w = crop.shape[:2]
                edge_pixels = np.concatenate([
                    crop[0, :].reshape(-1, 3),      # Top edge
                    crop[-1, :].reshape(-1, 3),     # Bottom edge
                    crop[:, 0].reshape(-1, 3),      # Left edge
                    crop[:, -1].reshape(-1, 3)      # Right edge
                ])
                all_edge_pixels.append(edge_pixels)
            
            # Combine all edge pixels
            all_grass_pixels = np.concatenate(all_edge_pixels, axis=0).astype(np.float32)
        
        # Convert grass pixels to LAB color space for better color analysis
        # Reshape to image format for cv2.cvtColor, then back to pixel array
        grass_pixels_bgr = all_grass_pixels.astype(np.uint8).reshape(1, -1, 3)
        grass_pixels_lab = cv2.cvtColor(grass_pixels_bgr, cv2.COLOR_BGR2LAB)
        all_grass_pixels_lab = grass_pixels_lab.reshape(-1, 3).astype(np.float32)
        
        # Calculate grass color statistics in LAB space
        grass_mean = np.mean(all_grass_pixels_lab, axis=0)
        grass_std = np.std(all_grass_pixels_lab, axis=0)

        # Define grass color range (mean ± 1.5 standard deviations)
        n_std_devs = 1.5
        grass_lower = grass_mean - n_std_devs * grass_std
        grass_upper = grass_mean + n_std_devs * grass_std

        # Ensure bounds are within valid color range [0, 255]
        grass_lower = np.clip(grass_lower, 0, 255)
        grass_upper = np.clip(grass_upper, 0, 255)
        
        kernel_open = _KERNEL_OPEN
        
        for i, crop in enumerate(crops):
            # Convert crop to LAB color space for consistent comparison
            crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Check if each pixel is within the grass color range for all channels
            grass_mask = np.all(
                (crop_lab >= grass_lower) & (crop_lab <= grass_upper), 
                axis=2
            ).astype(np.uint8)
            
            # Player mask is the inverse of grass mask
            player_mask = (1 - grass_mask).astype(np.uint8)
            
            # Clean the mask using morphological opening to remove noise
            cleaned_mask = cv2.morphologyEx(player_mask * 255, cv2.MORPH_OPEN, kernel_open)
            
            # Apply the cleaned mask to the original BGR crop
            masked_crop = crop.copy()
            masked_crop = cv2.bitwise_and(masked_crop, masked_crop, mask=cleaned_mask)
            masked_crop[cleaned_mask == 0] = [255, 255, 255]  # Set background to white
            masked_crops.append(masked_crop.copy())
        
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


