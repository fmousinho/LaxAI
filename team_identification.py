import logging
import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import Callable, Optional
import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
from videotools import BoundingBox
import utils 

logger = logging.getLogger(__name__)

class TeamIdentification:
    def __init__(self, player_feature_extractor: str = "dominant_color"):
        """Initializes the TeamIdentification class."""
        self.player_feature_extractor = player_feature_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._embedding_fn = None

        logger.info(f"Initializing TeamIdentification with feature extractor: {self.player_feature_extractor}")

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
            # This lambda ensures we get a (3,) shape vector or None from get_dominant_color
            self._embedding_fn = lambda roi_img: (
                (colors[0] if colors is not None and colors.shape[0] > 0 else None)
                if (colors := self.get_dominant_color(roi_img, k=1)) is not None else None
            )
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
        
        for frame, detections_in_frame in sampled_frames_data:
            for det_coords, _, _ in detections_in_frame:
                bbox = BoundingBox(*det_coords) 
                bbox_xyxy = bbox.to_xyxy()
                x1_roi, y1_roi, x2_roi, y2_roi = map(int, bbox_xyxy)
                roi = frame[y1_roi:y2_roi, x1_roi:x2_roi] 
                if roi.size == 0:
                    continue
                
                embedding = None
                if self._embedding_fn:
                    embedding = self._embedding_fn(roi)
                else: # Should ideally not be reached if __init__ sets a fallback
                    logger.error("Embedding function is not defined!")

                if embedding is not None:
                    player_roi_embeddings.append(embedding)
                    detection_bboxes.append(bbox)

        if len(player_roi_embeddings) < 2: # KMeans needs at least n_clusters (which is 2) samples
            logger.warning(f"Not enough valid player color ROIs collected ({len(player_roi_embeddings)}) to define two teams. Need at least 2. Defaulting team getter.")
            return _default_team_getter

        try:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(np.array(player_roi_embeddings))
            
            # Plotting - only if embeddings are 3D (like dominant color)
            embeddings_array_for_plot = np.array(player_roi_embeddings)
            if embeddings_array_for_plot.ndim == 2 and embeddings_array_for_plot.shape[1] == 3:
                utils.plot_team_kmeans_clusters(embeddings_array_for_plot, kmeans.labels_, kmeans.cluster_centers_)
            else:
                logger.info("Skipping 3D cluster plot as embeddings are not 3-dimensional (e.g., not RGB colors).")

            def get_team_id_for_bbox(bbox_to_check: BoundingBox, current_frame_for_roi: np.ndarray) -> Optional[int]:
                x1_roi, y1_roi, x2_roi, y2_roi = map(int, bbox_to_check.to_xyxy())
                frame_roi = current_frame_for_roi[y1_roi:y2_roi, x1_roi:x2_roi]
                if frame_roi.size == 0:
                    return None

                embedding_for_prediction = None
                if self._embedding_fn:
                    embedding_for_prediction = self._embedding_fn(frame_roi)
                
                if embedding_for_prediction is not None:
                    try:
                        return kmeans.predict(np.array([embedding_for_prediction]))[0]
                    except Exception as e:
                        logger.warning(f"KMeans prediction failed for embedding: {e}")
                        return None
                return None # If embedding is None or prediction fails
            return get_team_id_for_bbox
        
        except Exception as e:
            logger.error(f"Error during K-Means clustering for team definition: {e}", exc_info=True)
            return _default_team_getter
    
    def _get_dino_vit_embedding(self, roi_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts DINO ViT embedding for a given ROI image.
        roi_image is expected to be a NumPy array (H, W, C) in BGR format (from OpenCV).
        """
        if not self.dino_model or not self.dino_processor:
            logger.error("DINO ViT model or processor not initialized in _get_dino_vit_embedding.")
            return None
        
        try:
            # Convert BGR (OpenCV) to RGB then to PIL Image
            img_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
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

        top_crop_px = int(h * 0.10)
        bottom_crop_px = int(h * 0.40)
        left_crop_px = int(w * 0.20)
        right_crop_px = int(w * 0.20)

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
        # For green, 'a' should be less than ~128.
        # Example range for green grass in LAB (NEEDS CAREFUL TUNING!):
        # These values are for OpenCV's 0-255 scaled LAB.
        # 'a' channel values below 128 are greenish.
        lower_lab_grass = np.array([50, 0, 130])    # L_min, a_min (very green), b_min (yellowish)
        upper_lab_grass = np.array([200, 128, 200])  # L_max, a_max (still green, but less so), b_max (more yellowish)

        grass_mask = cv2.inRange(lab_image, lower_lab_grass, upper_lab_grass)
        non_grass_mask = cv2.bitwise_not(grass_mask)
        
        # Apply the mask to the original RGB image
        # Where mask is 0, the output will be black.
        masked_image_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=non_grass_mask)
        
        return masked_image_rgb

    def get_dominant_color(self, image: np.ndarray, k: int = 1) -> Optional[np.ndarray]:
        if image is None or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] != 3:
            logger.warning(f"get_dominant_color received an invalid image with shape: {image.shape if image is not None else 'None'}.")
            return None
        
        # Apply grass mask first (assuming 'image' is RGB)
        image_no_grass = self._apply_grass_mask(image)
        
        image_to_process = self._get_center_crop(image_no_grass)
        if image_to_process.size == 0: # If crop after mask is empty, try cropping original
            logger.warning(f"Center crop resulted in an empty image (orig_shape={image.shape}). Using full original image for resize.")
            image_to_process = image 
        try:
            resized_image = cv2.resize(image_to_process, (20, 20))
            if resized_image.shape[0] == 0 or resized_image.shape[1] == 0:
                logger.warning(f"Image became invalid after resizing in get_dominant_color. Original shape: {image.shape}")
                return None
        except cv2.error as e:
            logger.warning(f"cv2.resize error in get_dominant_color: {e}. Original image shape: {image.shape}")
            return None
        pixels = resized_image.reshape((-1, 3))
        if pixels.shape[0] < k: 
            logger.warning(f"Not enough pixels ({pixels.shape[0]}) to form {k} cluster(s) in get_dominant_color.")
            return None
        try:
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(pixels)
            return kmeans.cluster_centers_
        except Exception as e:
            logger.error(f"KMeans clustering failed in get_dominant_color: {e}", exc_info=True)
            return None