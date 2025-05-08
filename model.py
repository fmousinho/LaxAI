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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D scatter plot
from team_identification import TeamIdentification

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
        self.team_identifier = TeamIdentification() # Instantiate TeamIdentification

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
            bbox in xywh format
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
    
    def get_default_team_getter(self) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """Returns a default team getter function that always returns None."""
        return self.team_identifier.get_default_team_getter()

    def identifies_team (
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]]
    ) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """
        Delegates team identification to the TeamIdentification class.
        """
        return self.team_identifier.identifies_team(sampled_frames_data)