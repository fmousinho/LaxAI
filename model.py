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
        def _default_getter(_bbox: BoundingBox, _current_frame: np.ndarray) -> Optional[int]:
            return None
        return _default_getter

    def identifies_team (
        self,
        sampled_frames_data: list[tuple[np.ndarray, list]]
    ) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
        """
        Analyzes dominant colors of detected objects and groups them into two teams.

        Args:
            sampled_frames_data: A list of tuples, where each tuple contains:
                - frame (np.ndarray): An RGB video frame.
                - detections (list): A list of detections for that frame.
                  Each detection is (bbox_coords, confidence, class_id).

        Returns:
            A function that takes a BoundingBox object and the current frame as input
            and returns the team_id (0 or 1) or None.
        """
        _default_team_getter = self.get_default_team_getter()
        
        if not sampled_frames_data:
            logger.warning("No sampled frames data provided. Cannot define teams.")
            return _default_team_getter

        player_rois_colors = [] # To keep track of colors in each bbox bellow 
        detection_bboxes = [] # To keep track of bboxes corresponding to colors
        
        for frame, detections_in_frame in sampled_frames_data:
            for det_coords, _, _ in detections_in_frame:
                bbox = BoundingBox(*det_coords) # det_coords is [x1, y1, w, h]
                bbox_xyxy = bbox.to_xyxy()
                x1_roi, y1_roi, x2_roi, y2_roi = map(int, bbox_xyxy)
                roi = frame[y1_roi:y2_roi, x1_roi:x2_roi] # Use the specific frame for this detection
                if roi.size == 0:
                    continue
                
                dominant_color = self.get_dominant_color(roi)
                if dominant_color is not None:
                    player_rois_colors.append(dominant_color)
                    detection_bboxes.append(bbox) # Store the BoundingBox object itself
                else:
                    logger.warning(f"Could not determine dominant color for ROI from bbox {bbox} in a sampled frame. Skipping.")

        # Ensure we have enough valid colors for clustering into 2 teams
        if not player_rois_colors or len(player_rois_colors) < 2:
            logger.warning(f"Not enough players with valid dominant colors ({len(player_rois_colors)}) to define two teams. Need at least 2.")
            return _default_team_getter

        try:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(np.array(player_rois_colors))
            self._plot_team_kmeans_clusters(np.array(player_rois_colors), kmeans.labels_, kmeans.cluster_centers_)
            # _team_assignments_dict = {bbox: int(label) for bbox, label in zip(detection_bboxes, kmeans.labels_)}
            # logger.info(f"Team definition complete. Assignments: {_team_assignments_dict}")

            def get_team_id_for_bbox(bbox_to_check: BoundingBox, current_frame_for_roi: np.ndarray) -> Optional[int]:
                x1_roi, y1_roi, x2_roi, y2_roi = map(int, bbox_to_check.to_xyxy())
                frame_roi = current_frame_for_roi[y1_roi:y2_roi, x1_roi:x2_roi]
                if frame_roi.size == 0:
                    return None
                dominant_color = self.get_dominant_color(frame_roi)
                if dominant_color is not None:
                    try:
                        # Ensure dominant_color is a 2D array for predict
                        return kmeans.predict(np.array([dominant_color]))[0]
                    except Exception as e:
                        logger.warning(f"KMeans prediction failed for dominant color {dominant_color}: {e}")
                        return None
                return None
            return get_team_id_for_bbox
        
        except Exception as e:
            logger.error(f"Error during K-Means clustering for team definition: {e}", exc_info=True)
            return _default_team_getter
    
    def get_dominant_color(self, image: np.ndarray, k: int = 1) -> Optional[np.ndarray]:
        """
        Extracts the dominant color from an image region. Works with RGB and BRG formats.
        Assumes input `image` is in the format expected by KMeans (e.g., RGB if called from identifies_team).
        Returns None if the image is too small, invalid, or KMeans fails.

        Args:
            image: The input image as a numpy array
            k: Number of clusters for KMeans.

        Returns:
            The dominant color as a numpy array, or None on failure.
        """
        if image is None or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] != 3:
            logger.warning(f"get_dominant_color received an invalid image with shape: {image.shape if image is not None else 'None'}.")
            return None

        try:
            resized_image = cv2.resize(image, (20, 20))
            if resized_image.shape[0] == 0 or resized_image.shape[1] == 0: # Should not happen if input check is fine
                logger.warning(f"Image became invalid after resizing in get_dominant_color. Original shape: {image.shape}")
                return None
        except cv2.error as e:
            logger.warning(f"cv2.resize error in get_dominant_color: {e}. Original image shape: {image.shape}")
            return None

        pixels = resized_image.reshape((-1, 3))
        if pixels.shape[0] < k: # Need at least k samples for k clusters
            logger.warning(f"Not enough pixels ({pixels.shape[0]}) to form {k} cluster(s) in get_dominant_color.")
            return None
        try:
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(pixels)
            return kmeans.cluster_centers_[0]
        except Exception as e:
            logger.error(f"KMeans clustering failed in get_dominant_color: {e}", exc_info=True)
            return None
        
    def _plot_team_kmeans_clusters(self, colors_array: np.ndarray, labels: np.ndarray, centers: np.ndarray, output_filename: str = "team_kmeans_visualization.png"):
        """
        Generates and saves a 3D scatter plot of color clusters.

        Args:
            colors_array (np.ndarray): Array of RGB colors (N, 3).
            labels (np.ndarray): Cluster labels for each color.
            centers (np.ndarray): Cluster centers (n_clusters, 3).
            output_filename (str): Filename to save the plot.
        """
        if colors_array.shape[1] != 3:
            logger.warning("Cannot plot KMeans clusters: colors_array is not 3D (RGB).")
            return

        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot for player colors, colored by team label
            # Assuming labels are 0 and 1, map them to distinct colors for the plot
            plot_colors = ['blue', 'red'] # Team 0 will be blue, Team 1 will be red
            
            for i in range(len(colors_array)):
                ax.scatter(colors_array[i, 0], colors_array[i, 1], colors_array[i, 2], 
                           color=plot_colors[labels[i]], marker='o', alpha=0.6)

            # Plot cluster centers
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                       marker='x', s=200, color='black', label='Team Color Centers')

            ax.set_xlabel('Red Channel')
            ax.set_ylabel('Green Channel')
            ax.set_zlabel('Blue Channel')
            ax.set_title('KMeans Clustering of Player Dominant Colors for Teams')
            plt.legend()
            plt.savefig(output_filename)
            plt.close(fig) # Close the figure to free memory
            logger.info(f"KMeans team clustering visualization saved to {output_filename}")
        except Exception as e:
            logger.error(f"Error generating KMeans cluster plot: {e}", exc_info=True)
    