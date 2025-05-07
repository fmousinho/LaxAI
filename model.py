import logging
import io
import torch
import constants as const
import numpy as np
from store_driver import Store
from videotools import BoundingBox
from deep_sort_realtime.deepsort_tracker import DeepSort
from rfdetr import RFDETRBase # Import the custom class needed for loading
import cv2
from sklearn.cluster import KMeans
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class VideoModel:
    """Handles the loading and potentially inference of the video processing model."""

    def __init__(self, model_name: str,
                 drive_path: str, device: torch.device = None):
        """
        Initializes the VideoModel.

        Args:
            model_name: The name of the model file on Google Drive. Defaults to const.MODEL_NAME.
            drive_path: The path within Google Drive where the model file resides. Defaults to const.GOOGLE_DRIVE_PATH.
        """
        self.model = None  # Model will be loaded later
        self.model_name = model_name
        self.drive_path = drive_path
        self.device = device
        self.tracker = DeepSort(max_age=30)  # Initialize the tracker with a maximum age of 30 frames

    def load_from_drive(self, store: Store) -> bool:
        """
        Downloads the model file from Google Drive and loads it onto the specified device.

        Args:
            store: An initialized Store object for Google Drive access.
            device: The torch.device (cpu or cuda) to load the model onto.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        logger.info(f"Attempting to download model '{self.model_name}' from Google Drive path '{self.drive_path}'...")
        model_buffer = store.download_file_by_name(self.model_name, self.drive_path)

        if model_buffer:
            logger.info(f"Model file '{self.model_name}' downloaded successfully. Loading into PyTorch...")
            try:
                # Ensure custom classes needed for unpickling are added
                torch.serialization.add_safe_globals([const.MODEL_NAME])
                self.model = torch.load(model_buffer, map_location=self.device, weights_only=False)
                logger.info(f"Model loaded successfully to {self.device}")
                return True
            except Exception as e:
                logger.error(f"Error loading model from downloaded buffer: {e}", exc_info=True)
                self.model = None # Ensure model is None if loading fails
                return False
        else:
            logger.error(f"Failed to download model '{self.model_name}' from Google Drive. Cannot proceed.")
            self.model = None
            return False
        
    def generate_detections(self, frame: np.ndarray, threshold: float = 0.5) -> list:
        """
        Runs inference on a single frame using the loaded model.

        Args:
            frame: The input video frame as a numpy array.
            threshold: Confidence threshold for detections.

        Returns:
            A list of detected objects with their bounding boxes and scores.  
            [(bbox, confidence, class)] DeepSort expects
        """
        if self.model is None:
            logger.error("Model is not loaded. Cannot perform detection.")
            return []
        detections = self.model.predict(frame, device=self.device,threshold=threshold)

        detections_list = []
        for i in range(len(detections.xyxy)):
            # Only processes class_id 3 (player)
            if detections.class_id[i] !=3: continue  
            bbox = BoundingBox.from_xyxy(*detections.xyxy[i])
            detections_list.append(
                ([*bbox],
                detections.confidence[i],
                detections.class_id[i]))
     
        return detections_list
    
    def generate_tracks(self, frame: np.ndarray, detections: list) -> list:
        """
        Generates tracks for the detected objects using DeepSort.

        Args:
            frame: The input video frame as a numpy array.
            detections: A list of detected objects with their bounding boxes and scores.

        Returns:
            A list of tracked objects with their IDs and bounding boxes.
        """
        
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks
    
    def identifies_team (self, frame: np.ndarray, detections: list) -> Callable[[tuple], Optional[int]]:
        """
        Analyzes dominant colors of detected objects and groups them into two teams.

        Args:
            frame: The input video frame (expected in RGB format by get_dominant_color).
            detections: A list of detected objects. Each detection is expected to be
                        ([x1, y1, w, h], confidence, class_id).

        Returns:
            A function that takes a bounding box tuple (x1, y1, w, h) as input
            and returns the team_id (0 or 1) or None if not found or teams couldn't be defined.
        """
        
        def _default_team_getter(_bbox: tuple) -> Optional[int]:
            return None
        
        if not detections:
            logger.warning("No detections provided. Cannot define teams.")
            return _default_team_getter

        player_rois_colors = []
        detection_bboxes = [] # To keep track of bboxes corresponding to colors

        for det_bbox, _, _ in detections:
            x1, y1, x2, y2 = map(int, det_bbox.to_xyxy())
            roi = frame[y1:y2, x1:x2] # Frame is RGB, get_dominant_color will process RGB
            if roi.size == 0:
                continue
            
            dominant_color = self.get_dominant_color(roi) # Assuming get_dominant_color handles RGB
            player_rois_colors.append(dominant_color)
            detection_bboxes.append(tuple(det_bbox))

        if len(player_rois_colors) < 2: # Need at least 2 players/colors to form 2 teams
            logger.warning("Not enough players with valid ROIs to define two teams.")
            return {}

        try:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(player_rois_colors)



            # This dictionary is captured by the returned getter function
            _team_assignments_dict = {bbox: label for bbox, label in zip(detection_bboxes, kmeans.labels_)}
            logger.info(f"Team definition complete. Assignments: {_team_assignments_dict}")

            def get_team_id_for_bbox(bbox_to_check: BoundingBox) -> Optional[int]:
                x1, y1, x2, y2 = bbox_to_check.to_xyxy()

                return _team_assignments_dict.get(bbox_to_check)

            return get_team_id_for_bbox
        
        except Exception as e:
            logger.error(f"Error during K-Means clustering for team definition: {e}", exc_info=True)
            return _default_team_getter
    
    def get_dominant_color(self, image: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Extracts the dominant color from an image region.

        Args:
            image: The input image as a numpy array.
            k: Number of clusters for KMeans.

        Returns:
            The dominant color in BGR format.
        """
        image = cv2.resize(image, (20, 20))  # Speed-up
        pixels = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k).fit(pixels)
        return kmeans.cluster_centers_[0]

    
    