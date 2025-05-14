import logging
import numpy as np
import cv2
from typing import Callable, Optional
from .videotools import BoundingBox 
from . import utils
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

#DEFAULTS
FEATURE_EXTRACTOR = "dominant_color"
CLUSTERING_ALGORITHM = "kmeans"
GRASS_MASK = True
CENTER_CROP = True
RESIZE_DIMENSIONS = (20, 20)
TOP_CROP_PERCENTAGE = 0.1 #removes head
BOTTOM_CROP_PERCENTAGE = 0.4 #removes legs
LEFT_CROP_PERCENTAGE = 0.1
RIGHT_CROP_PERCENTAGE = 0.1


# Default LAB color ranges for grass masking (OpenCV's 0-255 scaled LAB)
# These will likely need tuning based on specific video conditions.
# 'a' channel values below 128 are greenish.
DEFAULT_LOWER_LAB_GRASS = np.array([60, 20, 130])    # L_min, a_min (very green), b_min (yellowish)
DEFAULT_UPPER_LAB_GRASS = np.array([200, 128, 200])  # L_max, a_max (still green, but less so), b_max (more yellowish)

class TeamIdentification:
    _kmeans_fail_counter = 0 # Class attribute to count KMeans failures for unique filenames
    def __init__(self, 
                 player_feature_extractor: str = FEATURE_EXTRACTOR,
                 clustering_algorithm: str = CLUSTERING_ALGORITHM,
                 grass_mask: bool = GRASS_MASK,
                 center_crop: bool = CENTER_CROP,
                 writer: Optional[SummaryWriter] = None):
        """Initializes the TeamIdentification class."""
        self.player_feature_extractor = player_feature_extractor
        self.clustering_algorithm = clustering_algorithm
        self.grass_mask = grass_mask
        self.center_crop = center_crop
        self.writer = writer

        logger.info(f"Initializing TeamIdentification with feature extractor: {self.player_feature_extractor}")

        self.writer = writer

        self.lower_lab_grass = DEFAULT_LOWER_LAB_GRASS
        self.upper_lab_grass = DEFAULT_UPPER_LAB_GRASS
        
        if self.player_feature_extractor == "dominant_color":
            # This embedding function returns a (1, 3) or (k, 3) color vector
            self._embedding_fn = self._get_dominant_color_embedding
        elif self.player_feature_extractor == "crop_no_grass":
            # This embedding function returns the processed image array itself
            self._embedding_fn = self._get_crop_no_grass_embedding
        else:
            logger.error(f"Unsupported player feature extractor: {self.player_feature_extractor}. Embedding function set to None.")
            self._embedding_fn = lambda: None

    def _get_dominant_color_embedding(self, roi_image: np.ndarray) -> Optional[np.ndarray]:
        """Helper to get a single dominant color as a (3,) embedding vector."""
        result = self._get_dominant_color(roi_image, k=1)
        dominant_colors = result
        embedding = dominant_colors[0] if dominant_colors is not None and dominant_colors.shape[0] > 0 else None
        return embedding

    def _get_crop_no_grass_embedding(self, roi_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Applies grass masking and center cropping to an ROI image.
        Returns the resulting image array as the "embedding".
        Compatible with the signature of _get_dominant_color_embedding.
        """
        if roi_image is None or roi_image.shape[0] == 0 or roi_image.shape[1] == 0 or roi_image.shape[2] != 3:
            logger.warning(f"_get_crop_no_grass_embedding received an invalid image with shape: {roi_image.shape if roi_image is not None else 'None'}.")
            return None

        image = self._apply_masks(roi_image)
        if image is None: return
      
        try:
            resized_embedding_img = cv2.resize(image, RESIZE_DIMENSIONS)
            embedding = resized_embedding_img.flatten().astype(np.float16)
        except cv2.error as e:
            logger.warning(f"cv2.error during resize/flatten in _get_crop_no_grass_embedding: {e}. Image shape: {image.shape}")
            return None

        return embedding

    def _apply_masks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Applies grass masking and center cropping to an image.
        Returns the processed image.
        """
        if self.grass_mask:
            image = self._apply_grass_mask(image)
        if self.center_crop:
            image = self._get_center_crop(image)
        if image is None or image.size == 0:
            logger.warning(f"Image pre-processing resulted in invalid image. Ignoring this ROI.")
            return None
        return image

    def get_default_team_getter(self) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """Returns a default team getter function that always returns None."""
        def _default_getter(_bbox: BoundingBox, _current_frame: np.ndarray) -> Optional[int]:
            return None
        return _default_getter

    def identifies_team (
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]],
    ) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """
        Analyzes embeddings of detected objects and groups them into two teams
        using the specified clustering algorithm.

        Args:
            sampled_frames_data: A list of tuples, where each tuple contains:
                - frame (np.ndarray): An RGB video frame.
                - detections (list): A list of detections for that frame.
                  Each detection is (bbox_coords, confidence, class_id). 

        Returns:
            A function that takes a BoundingBox object and the current frame as input
            and returns the team_id (0 or 1) or None.
        """
        if not sampled_frames_data:
            logger.warning("No sampled frames data provided. Cannot define teams.")
            return self._default_team_getter
        if self.clustering_algorithm == "kmeans":
            return self._kmeans_clustering(sampled_frames_data)
        else:
            logger.error(f"Unsupported clustering algorithm: {self.clustering_algorithm}. Defaulting team getter.")
            return self.get_default_team_getter()

    def _prepare_embeddings_for_clustering(
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]]
    ) -> Optional[np.ndarray]:
        """
        Collects, preprocesses, and normalizes embeddings from sampled frames.

        Returns:
            A NumPy array of normalized embeddings, or None if processing fails.
        """

        collected_embeddings = []

        for frame, detections_in_frame in sampled_frames_data:
            for det_coords, _, _ in detections_in_frame: # type: ignore
                bbox = BoundingBox(*det_coords)
                x1_roi, y1_roi, x2_roi, y2_roi = bbox.to_xyxy()
                roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
                if roi.size == 0: continue
                
                embedding = None
                if self._embedding_fn:
                    embedding = self._embedding_fn(roi) # Get embedding using the selected method
                
                if embedding is not None:
                    collected_embeddings.append(embedding)

        if len(collected_embeddings) < 2:
            logger.warning(f"Not enough embeddings collected ({len(collected_embeddings)}) for clustering. Need at least 2.")
            return None

        embeddings_array_f32 = np.array(collected_embeddings, dtype=np.float32)

        # Filter out NaNs or Infs from the NumPy array
        valid_mask = ~np.any(np.isnan(embeddings_array_f32), axis=1) & ~np.any(np.isinf(embeddings_array_f32), axis=1)
        embeddings_array_f32_filtered = embeddings_array_f32 # Start with the full array
     
        if not np.all(valid_mask):
            logger.warning("KMeans: Embeddings contain NaN/Inf values. Filtering them out.")
            embeddings_array_f32_filtered = embeddings_array_f32[valid_mask]

        if embeddings_array_f32_filtered.shape[0] < 2: 
            logger.warning(f"KMeans: Not enough valid (non-NaN/Inf) embeddings remaining ({embeddings_array_f32_filtered.shape[0]}) after filtering. Need at least 2 for KMeans. Defaulting team getter.")
            return None
        
        # Check for sufficient variance
        if embeddings_array_f32_filtered.shape[0] > 0 and np.all(np.std(embeddings_array_f32_filtered, axis=0) < 1e-5): # Safe to use .shape and np.std on NumPy array
                logger.warning(f"KMeans: Collected embeddings have near-zero variance (std: {np.std(embeddings_array_f32_filtered, axis=0)}). Cannot perform KMeans. Defaulting team getter.")
                return None

        # Normalize embeddings to [0, 1] range (assuming original data like colors/pixels is 0-255)
        normalized_embeddings_array = embeddings_array_f32_filtered / 255.0
        
        # Return the filtered and unnormalized embeddings as well, for TensorBoard visualization
        # The helper function will now return a tuple: (normalized_embeddings, unnormalized_filtered_embeddings)
        return normalized_embeddings_array, embeddings_array_f32_filtered

    def _kmeans_clustering(
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]]
    ) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """
        Performs team identification using KMeans clustering.
        """
        _default_team_getter = self.get_default_team_getter()
        
        processed_data = self._prepare_embeddings_for_clustering(sampled_frames_data)
        if processed_data is None:
            return _default_team_getter
        
        normalized_embeddings_array, embeddings_array_f32_filtered = processed_data
        kmeans_model = None

        try:
            with np.errstate(divide='raise', over='raise', invalid='raise'):
                logger.debug(f"KMeans: Attempting fit with normalized_embeddings_array shape: {normalized_embeddings_array.shape}, dtype: {normalized_embeddings_array.dtype}")
                kmeans_model = utils.CustomKMeans(n_clusters=2, random_state=42).fit(normalized_embeddings_array)
            
            logger.info("KMeans clustering for team definition completed successfully.")

            if self.writer:
                # Log embeddings to TensorBoard Projector
                # We use the unnormalized (but filtered) embeddings for visualization
                # And labels from KMeans
                # Ensure embeddings_array_f32_filtered is (N, D)
                if embeddings_array_f32_filtered.ndim == 2:
                    self.writer.add_embedding(embeddings_array_f32_filtered, 
                                              metadata=kmeans_model.labels_.tolist(),
                                              tag='player_embeddings_kmeans')
                if kmeans_model.cluster_centers_ is not None and kmeans_model.cluster_centers_.shape[1] == 3:
                    # Log cluster centers if they are 3D (e.g., colors)
                    # Scale them back to 0-255 for easier interpretation if they were colors
                    self.writer.add_image("KMeans_Team_Color_Centroids",
                                          (kmeans_model.cluster_centers_ * 255.0).astype(np.uint8).reshape(1, 2, 1, 3), # Reshape to (Batch=1, Height=2, Width=1, Channels=3) for NHWC
                                          dataformats='NHWC') # N=1, H=num_teams, W=1 (or num_features if not color)


            def get_team_id_for_bbox_kmeans(bbox_to_check: BoundingBox, current_frame_for_roi: np.ndarray) -> Optional[int]:
                if kmeans_model is None: return None # Should not happen if fit was successful
                
                x1, y1, x2, y2 = bbox_to_check.to_xyxy()
                roi = current_frame_for_roi[y1:y2, x1:x2]
                if roi.size == 0: return None
                
                embedding = self._embedding_fn(roi) 
                if embedding is None: return None
                
                embedding_np = np.array([embedding]) # Reshape to (1, D)
                if np.any(np.isnan(embedding_np)) or np.any(np.isinf(embedding_np)):
                    logger.warning("KMeans predict: Embedding for prediction contains NaN/Inf.")
                    return None
                normalized_embedding_for_prediction = embedding_np / 255.0
                
                return kmeans_model.predict(normalized_embedding_for_prediction)[0]
            
            return get_team_id_for_bbox_kmeans

        except Exception as e:
            logger.error(f"Error during KMeans clustering for team definition: {e}", exc_info=True)
            return _default_team_getter

    def _get_center_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts a crop of an image.

        Args:
            image: The input image as a numpy array.

        Returns:
            The cropped part of the image, or the original image if cropping
            results in an invalid (e.g., zero or negative dimension) region.
        """
        h, w = image.shape[:2]

        top_crop_px = int(h * TOP_CROP_PERCENTAGE)
        bottom_crop_px = int(h * BOTTOM_CROP_PERCENTAGE)
        left_crop_px = int(w * LEFT_CROP_PERCENTAGE)
        right_crop_px = int(w * RIGHT_CROP_PERCENTAGE)

        start_y = top_crop_px
        end_y = h - bottom_crop_px
        start_x = left_crop_px
        end_x = w - right_crop_px

        if start_x >= end_x or start_y >= end_y:
            logger.debug(f"Calculated crop dimensions are invalid (start_x={start_x}, end_x={end_x}, start_y={start_y}, end_y={end_y}). Original image h={h}, w={w}. Using full image.")
            return image
        
        return image[start_y:end_y, start_x:end_x]

    def _apply_grass_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """        
        Applies a mask to an RGB image to exclude common grass colors.
        Returns the masked RGB image.
        """
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

        grass_mask = cv2.inRange(lab_image, self.lower_lab_grass, self.upper_lab_grass)
        non_grass_mask = cv2.bitwise_not(grass_mask)
        
        # Apply the mask to the original RGB image
        # Where mask is 0, the output will be black.
        masked_image_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=non_grass_mask)
        
        return masked_image_rgb
    
    def _resize_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Resizes the image to the specified dimensions.
        Returns the resized image.
        """
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            logger.warning(f"resize_image received an invalid image with shape: {image.shape if image is not None else 'None'}.")
            return None
        
        try:
            resized_image = cv2.resize(image, RESIZE_DIMENSIONS)
            return resized_image
        except cv2.error as e:
            logger.warning(f"cv2.resize error in resize_image: {e}. Original image shape: {image.shape}")
            return None

    def _get_dominant_color(self, image: np.ndarray, k: int = 1) -> Optional[np.ndarray]:
        if image is None or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] != 3:
            logger.warning(f"get_dominant_color received an invalid image with shape: {image.shape if image is not None else 'None'}.")
            return None
        
        image = self._apply_masks(image)
        resized_image = self._resize_image(image)
        if resized_image is None: return None

        pixels = resized_image.reshape((-1, 3)) # Pixels are RGB

        # Pixels are np.uint8 at this point. NaN/Inf checks are not needed for uint8.

        # Filter out black pixels ([0,0,0])
        # Perform this check on the original uint8 pixels
        is_black_pixel = np.all(pixels == 0, axis=1)
        non_black_pixels_mask = ~is_black_pixel
        pixels_non_black = pixels[non_black_pixels_mask] # This will be uint8
        if pixels_non_black.shape[0] <k:
            logger.debug("Most pixels were black or became black after masking/cropping. No non-black pixels to process for dominant color.")
            return None

        try:
            with np.errstate(divide='raise', over='raise', invalid='raise'):
                # Convert to float32 specifically for KMeans
                pixels_f32 = pixels_non_black.astype(np.float32)
                # Normalize pixel values to [0, 1]
                normalized_pixels_for_kmeans = pixels_f32 / 255.0
                kmeans = utils.CustomKMeans(n_clusters=k, random_state=0).fit(normalized_pixels_for_kmeans)
                
                # Cluster centers are normalized ([0,1]), scale them back to [0,255]
                dominant_rgb_colors_normalized = kmeans.cluster_centers_
                dominant_rgb_colors_float = dominant_rgb_colors_normalized * 255.0
                dominant_rgb_colors = np.round(dominant_rgb_colors_float).astype(np.uint8)
                
                return dominant_rgb_colors
            
        except Exception as e:
            logger.error("KMeans clustering failed in the get_dominant_color method.")

            return None
