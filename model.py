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
            [(bbox list, confidence, class)] DeepSort expects
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

    def update_track_team_assignments(self, tracks: list, frame: np.ndarray, raw_team_id_getter: Callable[[BoundingBox, np.ndarray], Optional[int]]):
        """
        Updates the smoothed team assignments for each track based on history.

        Args:
            tracks: List of current tracks from the tracker.
            frame: The current video frame (RGB).
            raw_team_id_getter: The function that provides instantaneous team classification.
        """
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            current_raw_team_classification = None

            if raw_team_id_getter and hasattr(track, 'original_ltwh') and track.original_ltwh is not None:
                original_ltwh = track.original_ltwh
                original_detection_bbox = BoundingBox(x1=float(original_ltwh[0]),
                                                      y1=float(original_ltwh[1]),
                                                      w=float(original_ltwh[2]),
                                                      h=float(original_ltwh[3]))
                current_raw_team_classification = raw_team_id_getter(original_detection_bbox, frame)
            
            if track_id not in self.track_to_team_map:
                self.track_to_team_map[track_id] = {
                    'team_id': None,
                    'certainty': 0.0,
                    'classifications_history': collections.deque(maxlen=self.N_HISTORY_FRAMES),
                    'confirmed_team_id': None # Add a field for the "stickier" team ID
                }
            track_data = self.track_to_team_map[track_id]
            track_data['classifications_history'].append(current_raw_team_classification)
            
            valid_classifications_in_history = [cls for cls in track_data['classifications_history'] if cls is not None]

            new_assigned_team_id = None
            current_history_certainty = 0.0 # Certainty based on current history window
            count = collections.Counter() # Initialize count as an empty Counter

            if valid_classifications_in_history:
                count = collections.Counter(valid_classifications_in_history) # Populate if history exists
                if count: # True if valid_classifications_in_history was not empty
                    most_common_item = count.most_common(1)[0]
                    new_assigned_team_id = most_common_item[0]
                    current_history_certainty = most_common_item[1] / len(valid_classifications_in_history)

            # Logic for "sticky" team assignment
            old_confirmed_team = track_data['confirmed_team_id']
            
            if track_data['confirmed_team_id'] is None:
                # Try to establish an initial confirmed team
                if len(valid_classifications_in_history) >= self.MIN_HISTORY_FOR_CONFIRMATION and \
                   new_assigned_team_id is not None and \
                   current_history_certainty >= self.CONFIRMATION_THRESHOLD_CERTAINTY:
                    track_data['confirmed_team_id'] = new_assigned_team_id
                    track_data['team_id'] = new_assigned_team_id # Update the display team_id as well
                    track_data['certainty'] = current_history_certainty
                    logger.info(f"Track {track_id}: Initial team confirmed to {new_assigned_team_id} with certainty {current_history_certainty:.2f}")
                else:
                    # Not enough history or certainty yet for initial confirmation, use current smoothed
                    track_data['team_id'] = new_assigned_team_id 
                    track_data['certainty'] = current_history_certainty
            else:
                # We have a confirmed team, check if we should change it
                if new_assigned_team_id is not None and \
                   new_assigned_team_id != track_data['confirmed_team_id'] and \
                   current_history_certainty >= self.CONFIRMATION_THRESHOLD_CERTAINTY:
                    track_data['confirmed_team_id'] = new_assigned_team_id
                    track_data['team_id'] = new_assigned_team_id # Update the display team_id
                    track_data['certainty'] = current_history_certainty # Update certainty to reflect the new confirmation
                    logger.info(f"Track {track_id}: Confirmed team CHANGED to {new_assigned_team_id} (was {old_confirmed_team}) with certainty {current_history_certainty:.2f}")
                else:
                    # Stick with the confirmed_team_id for display, but update certainty based on current history
                    # This means 'team_id' (for display) remains the confirmed one,
                    # but 'certainty' reflects how well the current history supports that confirmed_team_id.
                    if track_data['confirmed_team_id'] in count: # Check if confirmed team is in current history counts
                        # If in count, valid_classifications_in_history was non-empty.
                        track_data['certainty'] = count[track_data['confirmed_team_id']] / len(valid_classifications_in_history)
                    else: # Confirmed team not in recent history, or history itself is empty (so count is empty)
                        track_data['certainty'] = 0.0 
                    track_data['team_id'] = track_data['confirmed_team_id'] # Ensure display team is the confirmed one

            if self.writer and track_data['team_id'] is not None: # Log certainty for assigned teams
                self.writer.add_scalar(f'TeamCertainty/Track_{track_id}', track_data['certainty'], global_step=logger.getEffectiveLevel()) # Use a global step, e.g., frame number
                # Note: global_step should ideally be the current frame index from the video processing loop.

    def get_smoothed_team_id_from_map(self, bbox: BoundingBox, current_frame: np.ndarray, track_id_for_lookup: str) -> Optional[int]:
        if track_id_for_lookup in self.track_to_team_map:
            return self.track_to_team_map[track_id_for_lookup].get('team_id')
        return None