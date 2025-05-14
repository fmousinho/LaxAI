import logging
import torch
from . import constants as const
import numpy as np
from .store_driver import Store
from .videotools import BoundingBox
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Callable, Optional
from .team_identification import TeamIdentification
import collections # For deque
import cv2 # For drawing on images
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class VideoModel:
    """Handles the loading and potentially inference of the video processing model."""

    def __init__(self, model_name: str,
                 drive_path: str, device: torch.device = None,
                 writer: Optional[SummaryWriter] = None):
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
        self.writer = writer
        self.tracker = DeepSort(max_age=30)  # Initialize the tracker with a maximum age of 30 frames
        self.team_identifier = TeamIdentification(writer=self.writer) # Instantiate TeamIdentification
        self.track_to_team_map = {} # For smoothed team assignments
        self.N_HISTORY_FRAMES = 20  # History window for smoothing
        self.MIN_HISTORY_FOR_CONFIRMATION = 10 # Min observations before a team can be "confirmed"
        self.CONFIRMATION_THRESHOLD_CERTAINTY = 0.75 # Min certainty to change a confirmed team or initially confirm

        # Constants for logging individual track ROIs
        self.LOG_TRACK_IMAGE_FRAME_INTERVAL = 10  # Log track ROIs every Nth call to update_track_team_assignments
        self.MAX_TRACK_ROIS_PER_LOGGED_FRAME = 3    # Max track ROIs to log from such a frame
        self._track_rois_logged_this_call = 0     # Internal counter

    def load_from_drive(self, store: Store) -> bool:
        """
        Downloads the model file from Google Drive and loads it onto the specified device.

        Args:
            store: An initialized Store object for Google Drive access.
            device: The torch.device (cpu or cuda) to load the model onto.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        logger.info(f"Attempting to download model '{self.model_name}' from Google Drive path '{self.drive_path}'...") # type: ignore[attr-defined]
        model_buffer = store.download_file_by_name(self.model_name, self.drive_path) # type: ignore[attr-defined]

        if model_buffer: # type: ignore[truthy-function]
            logger.info(f"Model file '{self.model_name}' downloaded successfully. Loading into PyTorch...")
            try:
                # Ensure custom classes needed for unpickling are added
                torch.serialization.add_safe_globals([const.MODEL_NAME]) # type: ignore[attr-defined]
                self.model = torch.load(model_buffer, map_location=self.device, weights_only=False)
                if self.writer: # Log new constants
                    config_tag_prefix = "VideoModel/Configuration" # Could be merged with TeamID if desired
                    self.writer.add_text(f"{config_tag_prefix}/TrackImageLogInterval", str(self.LOG_TRACK_IMAGE_FRAME_INTERVAL), 0)
                    self.writer.add_text(f"{config_tag_prefix}/MaxTrackROIsPerLoggedFrame", str(self.MAX_TRACK_ROIS_PER_LOGGED_FRAME), 0)
                    self.writer.add_text(f"{config_tag_prefix}/TrackHistoryFrames", str(self.N_HISTORY_FRAMES), 0)
                    self.writer.add_text(f"{config_tag_prefix}/TrackMinHistoryForConfirmation", str(self.MIN_HISTORY_FOR_CONFIRMATION), 0)
                    self.writer.add_text(f"{config_tag_prefix}/TrackConfirmationCertainty", str(self.CONFIRMATION_THRESHOLD_CERTAINTY), 0)
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
        
    def generate_detections(self, frame_in: np.ndarray, threshold: float = 0.5) -> list:
        """
        Runs inference on a single frame using the loaded model.

        Args:
            frame: The input video frame as a numpy array.
            threshold: Confidence threshold for detections.

        Returns:
            A list of detected objects with their bounding boxes and scores.  
            [(bbox list, confidence, class)] DeepSort expects
            bbox in xywh format
        """
        if self.model is None:
            logger.error("Model is not loaded. Cannot perform detection.")
            return []
        detections = self.model.predict(frame_in, device=self.device,threshold=threshold)

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
    
    def generate_tracks(self, frame_in: np.ndarray, detections: list) -> list:
        """
        Generates tracks for the detected objects using DeepSort.

        Args:
            frame: The input video frame as a numpy array.
            detections: A list of detected objects with their bounding boxes and scores.
        
        Returns:
            A list of tracked objects with their IDs and bounding boxes.
        """
        
        tracks = self.tracker.update_tracks(detections, frame=frame_in)
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

    def _get_raw_classification(self, track, frame_rgb: np.ndarray, raw_team_id_getter: Callable[[BoundingBox, np.ndarray], Optional[int]]) -> Optional[int]:
        """Gets the raw team classification for the current track and frame."""
        current_raw_team_classification = None
        if raw_team_id_getter and hasattr(track, 'original_ltwh') and track.original_ltwh is not None:
            original_ltwh = track.original_ltwh
            original_detection_bbox = BoundingBox(x1=float(original_ltwh[0]),
                                                  y1=float(original_ltwh[1]),
                                                  w=float(original_ltwh[2]),
                                                  h=float(original_ltwh[3]))
            current_raw_team_classification = raw_team_id_getter(original_detection_bbox, frame_rgb)
        return current_raw_team_classification

    def _update_classification_history(self, track_id: str, current_raw_team_classification: Optional[int]) -> dict:
        """Initializes or updates the classification history for a track."""
        if track_id not in self.track_to_team_map:
            self.track_to_team_map[track_id] = {
                'team_id': None,
                'certainty': 0.0,
                'classifications_history': collections.deque(maxlen=self.N_HISTORY_FRAMES),
                'confirmed_team_id': None
            }
        track_data = self.track_to_team_map[track_id]
        track_data['classifications_history'].append(current_raw_team_classification)
        return track_data

    def _calculate_current_assignment(self, track_data: dict) -> tuple[Optional[int], float, collections.Counter]:
        """Calculates the new potential team assignment and certainty based on history."""
        valid_classifications_in_history = [cls for cls in track_data['classifications_history'] if cls is not None]
        new_assigned_team_id = None
        current_history_certainty = 0.0
        count = collections.Counter()

        if valid_classifications_in_history:
            count = collections.Counter(valid_classifications_in_history)
            if count:
                most_common_item = count.most_common(1)[0]
                new_assigned_team_id = most_common_item[0]
                current_history_certainty = most_common_item[1] / len(valid_classifications_in_history)
        return new_assigned_team_id, current_history_certainty, count

    def _apply_sticky_assignment_logic(self, track_id: str, track_data: dict, new_assigned_team_id: Optional[int], current_history_certainty: float, history_counts: collections.Counter):
        """Applies sticky assignment logic to confirm or update team ID."""
        old_confirmed_team = track_data['confirmed_team_id']

        if track_data['confirmed_team_id'] is None:
            if len(track_data['classifications_history']) >= self.MIN_HISTORY_FOR_CONFIRMATION and \
               new_assigned_team_id is not None and \
               current_history_certainty >= self.CONFIRMATION_THRESHOLD_CERTAINTY:
                track_data['confirmed_team_id'] = new_assigned_team_id
                track_data['team_id'] = new_assigned_team_id
                track_data['certainty'] = current_history_certainty
                logger.info(f"Track {track_id}: Initial team confirmed to {new_assigned_team_id} with certainty {current_history_certainty:.2f}")
            else:
                track_data['team_id'] = new_assigned_team_id
                track_data['certainty'] = current_history_certainty
        else:
            if new_assigned_team_id is not None and \
               new_assigned_team_id != track_data['confirmed_team_id'] and \
               current_history_certainty >= self.CONFIRMATION_THRESHOLD_CERTAINTY:
                track_data['confirmed_team_id'] = new_assigned_team_id
                track_data['team_id'] = new_assigned_team_id
                track_data['certainty'] = current_history_certainty
                logger.info(f"Track {track_id}: Confirmed team CHANGED to {new_assigned_team_id} (was {old_confirmed_team}) with certainty {current_history_certainty:.2f}")
            else:
                if track_data['confirmed_team_id'] in history_counts:
                    track_data['certainty'] = history_counts[track_data['confirmed_team_id']] / len(track_data['classifications_history'])
                else:
                    track_data['certainty'] = 0.0
                track_data['team_id'] = track_data['confirmed_team_id']

    def _log_track_roi_to_tensorboard(self, track, track_data: dict, frame_rgb: np.ndarray, current_frame_index: int):
        """Logs an annotated ROI for the track to TensorBoard."""
        try:
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            frame_h, frame_w = frame_rgb.shape[:2]
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(frame_w, x2), min(frame_h, y2)

            if x1_c < x2_c and y1_c < y2_c:
                track_roi_rgb_crop = frame_rgb[y1_c:y2_c, x1_c:x2_c].copy()
                team_id_display = track_data['team_id']
                roi_tensor_chw_float_rgb = torch.from_numpy(track_roi_rgb_crop.copy()).permute(2, 0, 1).float() / 255.0
                tag = f"Track_Visuals/Frame_{current_frame_index}/Track_{track.track_id}_Team_{team_id_display if team_id_display is not None else 'NA'}"
                self.writer.add_image(tag, roi_tensor_chw_float_rgb, global_step=current_frame_index)
                self._track_rois_logged_this_call += 1
        except Exception as e:
            logger.error(f"Error logging track ROI for track {track.track_id} at frame {current_frame_index}: {e}", exc_info=True)

    def update_track_team_assignments(self, 
                                      tracks: list, 
                                      frame_rgb: np.ndarray, 
                                      raw_team_id_getter: Callable[[BoundingBox, np.ndarray], Optional[int]], # The getter expects RGB frame
                                      current_frame_index: int):
        """
        Updates the smoothed team assignments for each track based on history.
        Also logs individual track ROIs with team assignments to TensorBoard if in DEBUG mode.

        Args:
            tracks: List of current tracks from the tracker.
            frame_rgb: The current video frame (RGB).
            raw_team_id_getter: The function that provides instantaneous team classification.
            current_frame_index: The index of the current frame being processed.
        """
        is_logging_frame_for_track_rois = (current_frame_index % self.LOG_TRACK_IMAGE_FRAME_INTERVAL == 0)
        if is_logging_frame_for_track_rois:
            self._track_rois_logged_this_call = 0 # Reset counter for this frame # type: ignore[has-type]

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            
            # 1. Get raw classification
            current_raw_team_classification = self._get_raw_classification(track, frame_rgb, raw_team_id_getter)
            
            # 2. Update history
            track_data = self._update_classification_history(track_id, current_raw_team_classification)
            
            # 3. Calculate current assignment based on history
            new_assigned_team_id, current_history_certainty, history_counts = self._calculate_current_assignment(track_data)
            
            # 4. Apply sticky assignment logic
            self._apply_sticky_assignment_logic(track_id, track_data, new_assigned_team_id, current_history_certainty, history_counts)

            # Log certainty for assigned teams
            if self.writer and track_data['team_id'] is not None: 
                self.writer.add_scalar(f'TeamCertainty/Track_{track_id}', track_data['certainty'], global_step=current_frame_index)

            # Conditionally log track ROI
            if self.writer and logger.isEnabledFor(logging.DEBUG) and \
               is_logging_frame_for_track_rois and \
               self._track_rois_logged_this_call < self.MAX_TRACK_ROIS_PER_LOGGED_FRAME:
                self._log_track_roi_to_tensorboard(track, track_data, frame_rgb, current_frame_index)

    def get_smoothed_team_id_from_map(self, bbox: BoundingBox, current_frame_in: np.ndarray, track_id_for_lookup: str) -> Optional[int]:
        if track_id_for_lookup in self.track_to_team_map:
            return self.track_to_team_map[track_id_for_lookup].get('team_id')
        return None