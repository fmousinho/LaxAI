import logging
import numpy as np
import cv2
from typing import Callable, Optional
from .videotools import BoundingBox 
import torch 
import torchvision.utils # Import for make_grid
from . import utils
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

#DEFAULTS
FEATURE_EXTRACTOR = "crop_no_grass" # "dominant_color" or "crop_no_grass"
CLUSTERING_ALGORITHM = "kmeans"
GRASS_MASK = True
CENTER_CROP = True
RESIZE_DIMENSIONS = (20, 20)
TOP_CROP_PERCENTAGE = 0.1 #removes head
BOTTOM_CROP_PERCENTAGE = 0.4 #removes legs
LEFT_CROP_PERCENTAGE = 0.1
RIGHT_CROP_PERCENTAGE = 0.1
MAX_DEBUG_TEAM_ID_SAMPLE_FRAMES_TO_LOG = 10 # Log images from at most this many sampled frames
MAX_ROIS_PER_SAMPLED_FRAME_TO_LOG = 16     # Log at most this many ROIs per logged sampled frame

# Constants for learning grass color
MIN_GRASS_PIXELS_FOR_LEARNING = 50000  # Minimum number of pixels from bottom regions to attempt learning
GRASS_LEARNING_N_STD = 2.2           # Number of standard deviations to define the range around the mean
MAX_PIXELS_PER_FRAME_FOR_GRASS_LEARNING = 10000 # Max pixels to sample from each frame for grass learning
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

        if grass_mask:
            self.lower_lab_grass = DEFAULT_LOWER_LAB_GRASS
            self.upper_lab_grass = DEFAULT_UPPER_LAB_GRASS
        
        # Log configuration constants to TensorBoard if writer is available
        if self.writer:
            logger.debug("Logging TeamIdentification configuration to TensorBoard.")
            config_tag_prefix = "TeamIdentification/Configuration"
            self.writer.add_text(f"{config_tag_prefix}/FeatureExtractor", self.player_feature_extractor, 0)
            self.writer.add_text(f"{config_tag_prefix}/ClusteringAlgorithm", self.clustering_algorithm, 0)
            self.writer.add_text(f"{config_tag_prefix}/GrassMaskEnabled", str(self.grass_mask), 0)
            self.writer.add_text(f"{config_tag_prefix}/CenterCropEnabled", str(self.center_crop), 0)
            self.writer.add_text(f"{config_tag_prefix}/ResizeDimensions", str(RESIZE_DIMENSIONS), 0)
            self.writer.add_text(f"{config_tag_prefix}/TopCropPercentage", str(TOP_CROP_PERCENTAGE), 0)
            self.writer.add_text(f"{config_tag_prefix}/BottomCropPercentage", str(BOTTOM_CROP_PERCENTAGE), 0)
            self.writer.add_text(f"{config_tag_prefix}/LeftCropPercentage", str(LEFT_CROP_PERCENTAGE), 0)
            self.writer.add_text(f"{config_tag_prefix}/RightCropPercentage", str(RIGHT_CROP_PERCENTAGE), 0)
            self.writer.add_text(f"{config_tag_prefix}/MaxDebugSampleFramesToLog", str(MAX_DEBUG_TEAM_ID_SAMPLE_FRAMES_TO_LOG), 0)
            self.writer.add_text(f"{config_tag_prefix}/MaxROIsPerSampledFrameToLog", str(MAX_ROIS_PER_SAMPLED_FRAME_TO_LOG), 0)
            
        if self.player_feature_extractor == "dominant_color":
            # This embedding function returns a (1, 3) or (k, 3) color vector
            self._embedding_fn = self._get_dominant_color_embedding
        elif self.player_feature_extractor == "crop_no_grass":
            # This embedding function returns the processed image array itself
            self._embedding_fn = self._get_crop_no_grass_embedding
        else:
            logger.error(f"Unsupported player feature extractor: {self.player_feature_extractor}. Embedding function set to None.")
            self._embedding_fn = lambda roi: None

    def _learn_and_set_grass_parameters(self, sampled_frames_data_rgb: list[tuple[np.ndarray, list]]):
        """
        Learns grass color ranges from the bottom 2/3 of sampled frames and updates
        self.lower_lab_grass and self.upper_lab_grass.
        Logs the final grass parameters to TensorBoard.
        """
        logger.info("Attempting to learn grass color parameters from sampled frames...")
        all_l_values, all_a_values, all_b_values = [], [], []

        for frame_rgb, _ in sampled_frames_data_rgb:
            if frame_rgb is None or frame_rgb.ndim < 3 or frame_rgb.shape[2] != 3:
                logger.debug("Skipping a frame in grass learning due to invalid format.")
                continue

            h, w = frame_rgb.shape[:2]
            # Consider the bottom 2/3 of the frame for grass
            bottom_region_rgb = frame_rgb[h // 3:, :, :]

            if bottom_region_rgb.size == 0:
                logger.debug("Skipping a frame in grass learning due to empty bottom region.")
                continue

            # Reshape to a list of pixels for easier sampling
            all_bottom_pixels_rgb = bottom_region_rgb.reshape(-1, 3)
            num_available_pixels = all_bottom_pixels_rgb.shape[0]

            if num_available_pixels == 0:
                continue

            # Sub-sample pixels if there are too many
            sample_size = min(num_available_pixels, MAX_PIXELS_PER_FRAME_FOR_GRASS_LEARNING)
            sampled_indices = np.random.choice(num_available_pixels, size=sample_size, replace=False)
            sampled_pixels_rgb = all_bottom_pixels_rgb[sampled_indices, :]

            # Convert only the sampled RGB pixels to LAB
            sampled_pixels_lab = cv2.cvtColor(sampled_pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
            
            all_l_values.extend(sampled_pixels_lab[:, 0])
            all_a_values.extend(sampled_pixels_lab[:, 1])
            all_b_values.extend(sampled_pixels_lab[:, 2])

        if len(all_l_values) >= MIN_GRASS_PIXELS_FOR_LEARNING:
            l_mean, l_std = np.mean(all_l_values), np.std(all_l_values)
            a_mean, a_std = np.mean(all_a_values), np.std(all_a_values)
            b_mean, b_std = np.mean(all_b_values), np.std(all_b_values)

            self.lower_lab_grass = np.array([max(0, l_mean - GRASS_LEARNING_N_STD * l_std),
                                             max(0, a_mean - GRASS_LEARNING_N_STD * a_std),
                                             max(0, b_mean - GRASS_LEARNING_N_STD * b_std)], dtype=np.uint8)
            self.upper_lab_grass = np.array([min(255, l_mean + GRASS_LEARNING_N_STD * l_std),
                                             min(255, a_mean + GRASS_LEARNING_N_STD * a_std),
                                             min(255, b_mean + GRASS_LEARNING_N_STD * b_std)], dtype=np.uint8)
            logger.info(f"Successfully learned new grass LAB ranges. Lower: {self.lower_lab_grass}, Upper: {self.upper_lab_grass}")
        else:
            logger.warning(f"Not enough grass pixels ({len(all_l_values)}) found for learning. Using default LAB ranges.")
            self.lower_lab_grass = DEFAULT_LOWER_LAB_GRASS
            self.upper_lab_grass = DEFAULT_UPPER_LAB_GRASS

        if self.writer:
            config_tag_prefix = "Grass_Color/Configuration" # Same prefix as in __init__
            self.writer.add_text(f"{config_tag_prefix}/FinalLowerLABGrass", str(self.lower_lab_grass), 0)
            self.writer.add_text(f"{config_tag_prefix}/FinalUpperLABGrass", str(self.upper_lab_grass), 0)

            # Log images representing the determined LAB color bounds
            try:
                color_patch_size = (50, 50, 3) # Height, Width, Channels

                # Create image for lower LAB bound
                lower_lab_patch = np.full(color_patch_size, self.lower_lab_grass, dtype=np.uint8)
                lower_rgb_patch = cv2.cvtColor(lower_lab_patch, cv2.COLOR_LAB2RGB)
                lower_rgb_tensor = torch.from_numpy(lower_rgb_patch.copy()).permute(2, 0, 1).float() / 255.0
                self.writer.add_image("Grass_Color/LearnedGrassColor_LowerBound_RGB", lower_rgb_tensor, 0)

                # Create image for upper LAB bound
                upper_lab_patch = np.full(color_patch_size, self.upper_lab_grass, dtype=np.uint8)
                upper_rgb_patch = cv2.cvtColor(upper_lab_patch, cv2.COLOR_LAB2RGB)
                upper_rgb_tensor = torch.from_numpy(upper_rgb_patch.copy()).permute(2, 0, 1).float() / 255.0
                self.writer.add_image("Grass_Color/LearnedGrassColor_UpperBound_RGB", upper_rgb_tensor, 0)

                logger.info("Logged learned grass color bound images to TensorBoard.")

            except Exception as e:
                logger.error(f"Error logging learned grass color bound images to TensorBoard: {e}", exc_info=True)

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
      
        try: # This might be moved into _resize_image
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
        if self.grass_mask: # Consider merging these steps
            image = self._apply_grass_mask(image) 
        if image is None or image.size == 0:
            logger.warning(f"Image pre-processing resulted in invalid image. Ignoring this ROI.")
            return None
        if self.center_crop: # Consider merging these steps
            image = self._get_center_crop(image)
        if image is None or image.size == 0:
            logger.warning(f"Image pre-processing resulted in invalid image after center crop. Ignoring this ROI.")
            return None
        return image # Return explicitly

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
        if self.grass_mask: self._learn_and_set_grass_parameters(sampled_frames_data)
        if not sampled_frames_data:
            logger.warning("No sampled frames data provided. Cannot define teams.")
            return self.get_default_team_getter()
        if self.clustering_algorithm == "kmeans":
            return self._kmeans_clustering(sampled_frames_data)
        else:
            logger.error(f"Unsupported clustering algorithm: {self.clustering_algorithm}. Defaulting team getter.")
            return self.get_default_team_getter()

    def _prepare_roi_for_logging(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Applies standard pre-processing to an ROI for logging purposes.
        Uses a copy of the ROI to avoid modifying the original.
        """
        # Ensure we work on a copy for logging to avoid unintended modifications
        # to the ROI used for embedding calculation.
        roi_copy = roi.copy()
        
        processed_roi = self._apply_masks(roi_copy) 
        if processed_roi is None:
            # _apply_masks already logs a warning if it returns None
            return None
        
        resized_roi = self._resize_image(processed_roi)
        if resized_roi is None:
            # _resize_image already logs a warning if it returns None
            return None
        return resized_roi # HWC, RGB numpy array

    def _log_roi_grid_for_frame(self, 
                                rois_to_log_this_frame: list[np.ndarray], 
                                frame_idx_in_sample: int) -> None:
        """Logs a grid of pre-processed ROIs for a given frame to TensorBoard."""
        if not rois_to_log_this_frame:
            logger.debug(f"No valid ROIs provided to _log_roi_grid_for_frame for frame index {frame_idx_in_sample}.")
            return

        try:
            # Ensure all items are valid numpy arrays for tensor conversion
            # This check might be redundant if _prepare_roi_for_logging guarantees valid output or None
            valid_rois_for_tensor = [r for r in rois_to_log_this_frame if isinstance(r, np.ndarray) and r.ndim == 3 and r.shape[2] == 3]
            if not valid_rois_for_tensor:
                logger.debug(f"No valid numpy arrays in rois_to_log_this_frame for grid creation (frame index {frame_idx_in_sample}).")
                return

            tensor_rois = torch.stack([
                torch.from_numpy(img_np.copy()).permute(2, 0, 1).float() / 255.0
                for img_np in valid_rois_for_tensor
            ])
            
            image_grid = torchvision.utils.make_grid(tensor_rois, nrow=int(np.ceil(MAX_ROIS_PER_SAMPLED_FRAME_TO_LOG**0.5)))
            
            tag_name = f'TeamID_Sample_ROIs/Sampled_Frame_{frame_idx_in_sample}'
            self.writer.add_image(tag_name, image_grid, global_step=frame_idx_in_sample) 
        except Exception as e:
            logger.error(f"Error creating or logging image grid for frame index {frame_idx_in_sample}: {e}", exc_info=True)

    def _finalize_embeddings(self, collected_embeddings: list) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Converts, filters, checks variance, and normalizes collected embeddings."""
        if len(collected_embeddings) < 2:
            logger.warning(f"Not enough embeddings collected ({len(collected_embeddings)}) for clustering. Need at least 2.")
            return None

        embeddings_array_f32 = np.array(collected_embeddings, dtype=np.float32)
        valid_mask = ~np.any(np.isnan(embeddings_array_f32), axis=1) & ~np.any(np.isinf(embeddings_array_f32), axis=1)
        embeddings_array_f32_filtered = embeddings_array_f32
        if not np.all(valid_mask):
            logger.warning("Embeddings contain NaN/Inf values. Filtering them out.")
            embeddings_array_f32_filtered = embeddings_array_f32[valid_mask]
        if embeddings_array_f32_filtered.shape[0] < 2: 
            logger.warning(f"Not enough valid (non-NaN/Inf) embeddings remaining ({embeddings_array_f32_filtered.shape[0]}) after filtering. Need at least 2. Defaulting team getter.")
            return None
        if embeddings_array_f32_filtered.shape[0] > 0 and np.all(np.std(embeddings_array_f32_filtered, axis=0) < 1e-5):
            logger.warning(f"Collected embeddings have near-zero variance (std: {np.std(embeddings_array_f32_filtered, axis=0)}). Cannot perform clustering. Defaulting team getter.")
            return None
        normalized_embeddings_array = embeddings_array_f32_filtered / 255.0
        return normalized_embeddings_array, embeddings_array_f32_filtered

    def _prepare_embeddings_for_clustering(
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]]
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Collects, preprocesses, and normalizes embeddings from sampled frames.
        Also handles conditional logging of ROIs to TensorBoard if in DEBUG mode.
        Returns:
            A tuple of (normalized_embeddings_array, unnormalized_filtered_embeddings_array) or None.
        """
        collected_embeddings = []
        logged_sample_frames_count = 0 # For TensorBoard image logging limits

        should_log_images_for_team_id = self.writer is not None and logger.isEnabledFor(logging.DEBUG)
        if should_log_images_for_team_id:
            logger.debug(f"Team ID image logging is active. Will log up to {MAX_DEBUG_TEAM_ID_SAMPLE_FRAMES_TO_LOG} frames with up to {MAX_ROIS_PER_SAMPLED_FRAME_TO_LOG} ROIs each.")

        for frame_idx_in_sample, (frame, detections_in_frame) in enumerate(sampled_frames_data):
            rois_to_log_this_frame = []
            logged_rois_count_this_frame = 0
            
            for det_coords, _, _ in detections_in_frame: # type: ignore
                bbox = BoundingBox(*det_coords)
                x1_roi, y1_roi, x2_roi, y2_roi = bbox.to_xyxy()
                roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
                if roi.size == 0: continue

                embedding = None
                if self._embedding_fn:
                    embedding = self._embedding_fn(roi) if self._embedding_fn else None
                if embedding is not None:
                    collected_embeddings.append(embedding)

                # --- Prepare ROI for logging (if enabled and limits not met) ---
                if should_log_images_for_team_id and \
                   logged_sample_frames_count < MAX_DEBUG_TEAM_ID_SAMPLE_FRAMES_TO_LOG and \
                   logged_rois_count_this_frame < MAX_ROIS_PER_SAMPLED_FRAME_TO_LOG:
                    
                    prepared_roi_for_log = self._prepare_roi_for_logging(roi)
                    if prepared_roi_for_log is not None:
                        rois_to_log_this_frame.append(prepared_roi_for_log)
                        logged_rois_count_this_frame += 1
            
            # --- Log ROI Grid for the current frame (if enabled and limits not met) ---
            if should_log_images_for_team_id and \
               logged_sample_frames_count < MAX_DEBUG_TEAM_ID_SAMPLE_FRAMES_TO_LOG and \
               rois_to_log_this_frame:
                self._log_roi_grid_for_frame(rois_to_log_this_frame, frame_idx_in_sample)
                logged_sample_frames_count += 1 # Increment count as a grid for this frame was attempted/logged

        # --- Finalize and return embeddings ---
        finalized_embeddings = self._finalize_embeddings(collected_embeddings)
        return finalized_embeddings

    def _kmeans_clustering(
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]]
    ) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """
        Trains a KMeans model on the embeddings of sampled frames to identify teams

        Returns a function that uses the trained KMeans model to assign team IDs
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
                                              metadata=kmeans_model.labels_.tolist() if kmeans_model.labels_ is not None else None,
                                              tag='player_embeddings_kmeans')
                if kmeans_model.cluster_centers_ is not None and kmeans_model.cluster_centers_.shape[1] == 3:
                    # Log cluster centers if they are 3D (e.g., colors)
                    # Scale them back to 0-255 for easier interpretation if they were colors
                    self.writer.add_image("KMeans_Team_Color_Centroids",
                                          (kmeans_model.cluster_centers_ * 255.0).astype(np.uint8).reshape(1, 2, 1, 3), # Reshape to (Batch=1, Height=2, Width=1, Channels=3) for NHWC
                                          dataformats='NHWC', global_step=0) # N=1, H=num_teams, W=1 (or num_features if not color)


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
        
        masked_image = self._apply_masks(image)
        if masked_image is None:
            logger.warning("Image masking resulted in None. Skipping further processing.")
            return None
        image = masked_image
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
                if dominant_rgb_colors_normalized is None:
                    logger.error("KMeans clustering returned None for cluster centers.")
                    return None
                dominant_rgb_colors_float = dominant_rgb_colors_normalized * 255.0
                dominant_rgb_colors = np.round(dominant_rgb_colors_float).astype(np.uint8)
                
                return dominant_rgb_colors
            
        except Exception as e:
            logger.error("KMeans clustering failed in the get_dominant_color method.")

            return None
