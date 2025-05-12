import logging
import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import Callable, Optional, Union, Dict
import torch # Added missing import
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
from .videotools import BoundingBox, VideoToools 
from . import utils
import base64 # For encoding images for HTML report
import io # For image encoding
import os # For path manipulation

logger = logging.getLogger(__name__)

# Default LAB color ranges for grass masking (OpenCV's 0-255 scaled LAB)
# These will likely need tuning based on specific video conditions.
# 'a' channel values below 128 are greenish.
DEFAULT_LOWER_LAB_GRASS = np.array([60, 20, 130])    # L_min, a_min (very green), b_min (yellowish)
DEFAULT_UPPER_LAB_GRASS = np.array([200, 128, 200])  # L_max, a_max (still green, but less so), b_max (more yellowish)

class TeamIdentification:
    _kmeans_fail_counter = 0 # Class attribute to count KMeans failures for unique filenames
    _avg_hsv_debug_save_counter = 0 # Class attribute for new get_dominant_color debug images
    def __init__(self, player_feature_extractor: str = "dominant_color"):
        """Initializes the TeamIdentification class."""
        self.player_feature_extractor = player_feature_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._embedding_fn = None

        logger.info(f"Initializing TeamIdentification with feature extractor: {self.player_feature_extractor}")

        # Make grass mask parameters configurable if needed, or use defaults
        self.lower_lab_grass = DEFAULT_LOWER_LAB_GRASS
        self.upper_lab_grass = DEFAULT_UPPER_LAB_GRASS

        if self.player_feature_extractor == "dino_vit":
            self.dino_processor = None
            self.dino_model = None
            self._initialize_dino_vit_model() # This sets self.dino_model and self.dino_processor
            if self.dino_model and self.dino_processor:
                self._embedding_fn = self._get_dino_vit_embedding
            else:
                logger.error("DINO ViT model failed to initialize. Embedding function will be non-functional for DINO.")
                self._embedding_fn = lambda roi_img: None # Fallback
        elif self.player_feature_extractor == "dominant_color":
            self._embedding_fn = self._get_dominant_color_embedding
        else:
            logger.error(f"Unsupported player feature extractor: {self.player_feature_extractor}. Embedding function set to None.")
            self._embedding_fn = lambda roi_img: None # Fallback for unsupported types

    def _initialize_dino_vit_model(self):
        try:
            self.dino_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
            self.dino_model = ViTModel.from_pretrained('facebook/dino-vitb16').to(self.device).eval()
            logger.info(f"DINO ViT model (facebook/dino-vitb16) loaded to {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load DINO ViT model: {e}", exc_info=True)
            self.dino_processor = None
            self.dino_model = None

    def _get_dominant_color_embedding(self, roi_image: np.ndarray, return_intermediate_images: bool = False) -> Union[tuple[Optional[np.ndarray], Optional[Dict]], Optional[np.ndarray]]:
        """Helper to get a single dominant color as a (3,) embedding vector."""
        # Always apply grass mask for embedding logic; report will reflect this.
        result = self.get_dominant_color(roi_image, k=1, apply_grass_mask=True, return_intermediate_images=return_intermediate_images)

        if return_intermediate_images:
            # result is (dominant_colors, intermediate_dict) or (None, intermediate_dict)
            dominant_colors, intermediates = result if isinstance(result, tuple) and len(result) == 2 else (result, {})
            embedding = dominant_colors[0] if dominant_colors is not None and dominant_colors.shape[0] > 0 else None
            return embedding, intermediates
        else:
            # result is dominant_colors or None
            dominant_colors = result
            embedding = dominant_colors[0] if dominant_colors is not None and dominant_colors.shape[0] > 0 else None
            return embedding


    def get_default_team_getter(self) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """Returns a default team getter function that always returns None."""
        def _default_getter(_bbox: BoundingBox, _current_frame: np.ndarray) -> Optional[int]:
            return None
        return _default_getter

    def identifies_team (
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]],
        clustering_algorithm: str = "kmeans" # New argument with a default
    ) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """
        Analyzes embeddings detected objects and groups them into two teams
        using the specified clustering algorithm.

        Args:
            sampled_frames_data: A list of tuples, where each tuple contains:
                - frame (np.ndarray): An RGB video frame.
                - detections (list): A list of detections for that frame.
                  Each detection is (bbox_coords, confidence, class_id).
            clustering_algorithm (str): The name of the clustering algorithm to use.
                                        Currently supports "kmeans".

        Returns:
            A function that takes a BoundingBox object and the current frame as input
            and returns the team_id (0 or 1) or None.
        """
        if clustering_algorithm == "kmeans":
            return self._kmeans_only_clustering(sampled_frames_data)
        else:
            logger.warning(f"Unsupported clustering algorithm: {clustering_algorithm}. Defaulting team getter.")
            return self.get_default_team_getter()

    def _kmeans_only_clustering(
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]]
    ) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """
        Performs team identification using KMeans clustering.
        """
        _default_team_getter = self.get_default_team_getter()
        if not sampled_frames_data:
            logger.warning("No sampled frames data provided. Cannot define teams.")
            return _default_team_getter

        detection_bboxes = [] 
        player_roi_embeddings = []
        html_table_rows = []

        # Generate ROI report only if logger is DEBUG and using dominant color
        generate_roi_report = (logger.getEffectiveLevel() <= logging.DEBUG and
                               self.player_feature_extractor == "dominant_color")
        
        for frame, detections_in_frame in sampled_frames_data:
            for det_coords, _, _ in detections_in_frame:
                bbox = BoundingBox(*det_coords) 
                bbox_xyxy = bbox.to_xyxy()
                x1_roi, y1_roi, x2_roi, y2_roi = map(int, bbox_xyxy)
                # Ensure ROI coordinates are valid
                if x1_roi >= x2_roi or y1_roi >= y2_roi:
                    logger.debug(f"Skipping invalid ROI coordinates: {bbox_xyxy}")
                    continue
                roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
                if roi.size == 0:
                    logger.debug(f"Skipping empty ROI for bbox: {bbox_xyxy}")
                    continue
                
                embedding = None
                intermediate_data_for_report = None

                if self._embedding_fn:
                    if self._embedding_fn == self._get_dominant_color_embedding:
                        # Pass generate_roi_report to control if intermediate images are returned
                        result_from_embedding_fn = self._embedding_fn(roi, return_intermediate_images=generate_roi_report)
                        if generate_roi_report:
                            embedding, intermediate_data_for_report = result_from_embedding_fn
                        else:
                            embedding = result_from_embedding_fn
                    else: # For DINO or other extractors not producing this specific report
                        embedding = self._embedding_fn(roi)
                else: # Should ideally not be reached if __init__ sets a fallback
                    logger.error("Embedding function is not defined!")

                if embedding is not None:
                    player_roi_embeddings.append(embedding)
                    detection_bboxes.append(bbox)

                    if generate_roi_report and intermediate_data_for_report and self.player_feature_extractor == "dominant_color":
                        html_row = utils.format_roi_for_html_report(intermediate_data_for_report, embedding) # embedding is the dominant color here
                        if html_row:
                            html_table_rows.append(html_row)

        if len(player_roi_embeddings) < 2: # KMeans needs at least n_clusters (which is 2) samples
            logger.warning(f"Not enough player ROIs collected ({len(player_roi_embeddings)}) to attempt team definition. Need at least 2. Defaulting team getter.")
            return _default_team_getter

        # Convert to numpy array for easier manipulation and checking
        embeddings_array = np.array(player_roi_embeddings, dtype=np.float16) # Specify dtype for consistency

        # Check for NaNs or Infs in embeddings and filter them out
        valid_mask = ~np.any(np.isnan(embeddings_array), axis=1) & ~np.any(np.isinf(embeddings_array), axis=1)
        if not np.all(valid_mask):
            logger.warning(f"Embeddings contain NaN or Inf values. Filtering them out. Original count: {len(embeddings_array)}")
            embeddings_array = embeddings_array[valid_mask]
            # Note: detection_bboxes would also need filtering if used directly with filtered embeddings later.
            # For now, the primary concern is the KMeans input.
            logger.info(f"Proceeding with {embeddings_array.shape[0]} valid embeddings after filtering NaN/Inf.")

        if embeddings_array.shape[0] < 2: # Check again after filtering
            logger.warning(f"Not enough valid (non-NaN/Inf) embeddings remaining ({embeddings_array.shape[0]}) after filtering. Need at least 2 for KMeans. Defaulting team getter.")
            return _default_team_getter

        try:
            # Temporarily set NumPy to raise exceptions for these warnings
            with np.errstate(divide='raise', over='raise', invalid='raise'):
                logger.debug(f"Attempting KMeans.fit with embeddings_array of shape: {embeddings_array.shape}, dtype: {embeddings_array.dtype}")
                # logger.debug(f"Sample of embeddings_array before fit: {embeddings_array[:5]}") # Uncomment to log sample data
                kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(embeddings_array)

            # Plotting - only if embeddings are 3D (like dominant color)
            # Use the cleaned embeddings_array for plotting
            if embeddings_array.ndim == 2 and embeddings_array.shape[1] == 3:
                utils.plot_team_kmeans_clusters(embeddings_array, kmeans.labels_, kmeans.cluster_centers_)
            else:
                logger.info("Skipping 3D cluster plot as embeddings are not 3-dimensional (e.g., not RGB colors).")

            def get_team_id_for_bbox(bbox_to_check: BoundingBox, current_frame_for_roi: np.ndarray) -> Optional[int]:
                x1_roi, y1_roi, x2_roi, y2_roi = map(int, bbox_to_check.to_xyxy())
                frame_roi = current_frame_for_roi[y1_roi:y2_roi, x1_roi:x2_roi]
                if frame_roi.size == 0:
                    return None

                embedding_for_prediction = None
                if self._embedding_fn:
                    # For prediction, we don't need intermediate images, so pass False
                    if self._embedding_fn == self._get_dominant_color_embedding:
                         embedding_for_prediction = self._embedding_fn(frame_roi, return_intermediate_images=False)
                    else:
                         embedding_for_prediction = self._embedding_fn(frame_roi)
                
                if embedding_for_prediction is not None:
                    # Check the single embedding for prediction as well
                    embedding_np = np.array([embedding_for_prediction], dtype=np.float16)
                    if np.any(np.isnan(embedding_np)) or np.any(np.isinf(embedding_np)):
                        logger.warning(f"Embedding for prediction contains NaN/Inf. Cannot predict team for this ROI.")
                        return None
                    try:
                        with np.errstate(divide='raise', over='raise', invalid='raise'):
                            # logger.debug(f"Attempting KMeans.predict with embedding_np: {embedding_np}") # Uncomment for predict data
                            return kmeans.predict(embedding_np)[0]
                    except Exception as e:
                        # This will now catch the raised exceptions from np.errstate
                        logger.warning(f"KMeans prediction failed for ROI embedding: {e}")
                        return None
                return None # If embedding is None or prediction fails
            
            # After the loop and successful KMeans, generate the HTML report if data was collected
            if generate_roi_report and html_table_rows:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                report_output_dir = os.path.join(script_dir, "debug_team_identification_rois")
                os.makedirs(report_output_dir, exist_ok=True)
                utils.generate_and_save_roi_report_html(html_table_rows, report_output_dir)
                logger.info(f"Team ID ROI processing HTML report saved in: {report_output_dir}")

            return get_team_id_for_bbox
        
        except Exception as e:
            logger.error(f"Error during K-Means clustering for team definition: {e}", exc_info=True)
            return _default_team_getter
    
    def _get_dino_vit_embedding(self, roi_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts DINO ViT embedding for a given ROI image.
        roi_image is expected to be a NumPy array (H, W, C) in RGB format.
        """
        if not self.dino_model or not self.dino_processor:
            logger.error("DINO ViT model or processor not initialized in _get_dino_vit_embedding.")
            return None
        
        try:
            # Assuming roi_image is already RGB, directly convert to PIL Image
            img_rgb = roi_image 
            pil_image = Image.fromarray(img_rgb)

            inputs = self.dino_processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            
            # For DINO ViT, the [CLS] token embedding is often used as the global image representation.
            # The [CLS] token is typically the first token in the sequence.
            cls_token_embedding = outputs.last_hidden_state[:, 0, :] # Batch_size, CLS_token_index, Hidden_dim
            
            return cls_token_embedding.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Error getting DINO ViT embedding: {e}", exc_info=True)
            return None

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

        top_crop_px = int(h * 0.1)
        bottom_crop_px = int(h * 0.4)
        left_crop_px = int(w * 0.1)
        right_crop_px = int(w * 0.1)

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

        # --- WILL REQUIRE CAREFUL TUNING for your specific videos. ---
        # L: Lightness (0-100, but OpenCV scales it to 0-255 for 8-bit images)
        # a: Green-Red axis (approx -128 to 127, OpenCV maps to 0-255, where ~128 is neutral)
        # b: Blue-Yellow axis (approx -128 to 127, OpenCV maps to 0-255, where ~128 is neutral)
        grass_mask = cv2.inRange(lab_image, self.lower_lab_grass, self.upper_lab_grass)
        non_grass_mask = cv2.bitwise_not(grass_mask)
        
        # Apply the mask to the original RGB image
        # Where mask is 0, the output will be black.
        masked_image_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=non_grass_mask)
        
        return masked_image_rgb

    def get_dominant_color(self, image: np.ndarray, k: int = 1, apply_grass_mask: bool = False, return_intermediate_images: bool = False) -> Union[tuple[Optional[np.ndarray], Dict], Optional[np.ndarray]]:
        if image is None or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] != 3:
            logger.warning(f"get_dominant_color received an invalid image with shape: {image.shape if image is not None else 'None'}.")
            return (None, {}) if return_intermediate_images else None
        
        intermediate_results = {}
        if return_intermediate_images:
            intermediate_results['original_roi'] = image.copy()

        # Process for masking
        image_for_masking = image.copy() # Use a copy for masking step
        if apply_grass_mask:
            logger.debug("Applying grass mask in get_dominant_color.")
            image_after_mask = self._apply_grass_mask(image_for_masking)
            if return_intermediate_images:
                intermediate_results['masked_roi'] = image_after_mask.copy()
        else:
            image_after_mask = image_for_masking # No mask applied
            if return_intermediate_images:
                # If no mask applied, "masked" is same as original for reporting
                intermediate_results['masked_roi'] = image_after_mask.copy() 
        
        # Process for cropping (applied to the result of masking step)
        image_after_crop = self._get_center_crop(image_after_mask)
        if image_after_crop.size == 0:
            logger.debug(f"Center crop resulted in an empty image (orig_shape={image.shape}, after_mask_shape={image_after_mask.shape}). Using image before crop.")
            image_after_crop = image_after_mask # Fallback to image before crop
            if image_after_crop.size == 0: # If image_after_mask was also empty (e.g. fully masked out)
                 logger.warning(f"Image after mask was empty. Using original image for dominant color.")
                 image_after_crop = image.copy() # Ultimate fallback to original

        if return_intermediate_images:
            intermediate_results['cropped_roi'] = image_after_crop.copy()

        # Final image to process for dominant color (after mask and crop)
        final_image_for_kmeans = image_after_crop

        try:
            # Ensure final_image_for_kmeans is valid before resize
            if final_image_for_kmeans is None or final_image_for_kmeans.size == 0:
                logger.warning(f"Final image for K-Means is empty or None before resize. Shape: {final_image_for_kmeans.shape if final_image_for_kmeans is not None else 'None'}")
                return (None, intermediate_results) if return_intermediate_images else None

            resized_image = cv2.resize(final_image_for_kmeans, (20, 20)) # This is RGB
            # Ensure resized_image is not empty and has 3 channels
            if resized_image.shape[0] == 0 or resized_image.shape[1] == 0:
                logger.warning(f"Image became invalid after resizing in get_dominant_color. Original shape: {image.shape}")
                return (None, intermediate_results) if return_intermediate_images else None
            
        except cv2.error as e:
            logger.warning(f"cv2.resize error in get_dominant_color: {e}. Original image shape: {image.shape}")
            return (None, intermediate_results) if return_intermediate_images else None

        pixels = resized_image.reshape((-1, 3)) # Pixels are RGB

        # Check for NaN/Inf in pixels (e.g. if image_to_process had issues)
        if np.any(np.isnan(pixels)) or np.any(np.isinf(pixels)):
            logger.warning(f"Pixel data contains NaN/Inf values BEFORE astype in get_dominant_color. Pixels: {pixels[:5]}")
            return (None, intermediate_results) if return_intermediate_images else None

        pixels_float32 = pixels.astype(np.float32)

        if np.any(np.isnan(pixels_float32)) or np.any(np.isinf(pixels_float32)):
            logger.warning(f"Pixel data contains NaN/Inf values AFTER astype in get_dominant_color. Pixels: {pixels_float32[:5]}")
            return (None, intermediate_results) if return_intermediate_images else None

        # Filter out black pixels ([0,0,0])
        is_black_pixel = np.all(pixels_float32 == 0, axis=1)
        non_black_pixels_mask = ~is_black_pixel
        pixels_float32_non_black = pixels_float32[non_black_pixels_mask]

        if pixels_float32_non_black.shape[0] == 0:
            logger.debug("All pixels were black or became black after masking/cropping. No non-black pixels to process for dominant color.")
            return (None, intermediate_results) if return_intermediate_images else None
        
        # Update pixels_float32 to be the non-black version for subsequent processing
        pixels_float32 = pixels_float32_non_black

        # Check for sufficient variance if all remaining non-black pixels are nearly identical
        # The pixels_float32.shape[0] > 0 check is implicitly true here due to the previous check for empty pixels_float32_non_black
        if np.all(np.std(pixels_float32, axis=0) < 1e-5): # Small threshold for variance
            logger.debug(f"Remaining non-black RGB pixels have near-zero variance (std: {np.std(pixels_float32, axis=0)}). Using average of these non-black pixels.")
            avg_rgb_color = np.mean(pixels_float32, axis=0, keepdims=True)
            if return_intermediate_images:
                return avg_rgb_color, intermediate_results
            else:
                return avg_rgb_color

        if pixels_float32.shape[0] < k:
            logger.warning(f"Not enough non-black RGB pixels ({pixels_float32.shape[0]}) to form {k} cluster(s) in get_dominant_color.")
            return (None, intermediate_results) if return_intermediate_images else None
        try:
            with np.errstate(divide='raise', over='raise', invalid='raise'):
                # KMeans on RGB pixels
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(pixels_float32) 
                dominant_rgb_colors = kmeans.cluster_centers_ # Cluster centers are already RGB
                
                if return_intermediate_images:
                    return dominant_rgb_colors, intermediate_results
                else:
                    return dominant_rgb_colors
        except Exception as e:
            TeamIdentification._kmeans_fail_counter += 1
            error_msg = (f"KMeans clustering failed in get_dominant_color (attempt {TeamIdentification._kmeans_fail_counter}). "
                         f"Error: {e}. Input pixels (first 5 of {pixels_float32.shape[0]}): {pixels_float32[:5]}")
            
            # Save problematic images for visual inspection
            try:
                debug_image_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_kmeans_fail_images")
                os.makedirs(debug_image_save_dir, exist_ok=True)

                if 'final_image_for_kmeans' in locals() and final_image_for_kmeans is not None and final_image_for_kmeans.size > 0:
                    pre_resize_path = os.path.join(debug_image_save_dir, f"fail_kmeans_input_{TeamIdentification._kmeans_fail_counter}_pre_resize.png")
                    cv2.imwrite(pre_resize_path, cv2.cvtColor(final_image_for_kmeans, cv2.COLOR_RGB2BGR))
                    error_msg += f" Saved pre-resize image to {pre_resize_path}."
                if 'resized_image' in locals() and resized_image is not None and resized_image.size > 0:
                    resized_path = os.path.join(debug_image_save_dir, f"fail_kmeans_input_{TeamIdentification._kmeans_fail_counter}_resized.png")
                    cv2.imwrite(resized_path, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
                    error_msg += f" Saved resized image to {resized_path}."
            except Exception as save_exc:
                error_msg += f" Additionally, failed to save debug images: {save_exc}."
            
            logger.error(error_msg, exc_info=True)
            return (None, intermediate_results) if return_intermediate_images else None
