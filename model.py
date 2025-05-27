import logging
import torch
from . import constants as const
import numpy as np
from .store_driver import Store
from .detection import DetectionModel  # Import the new DetectionModel
from .videotools import BoundingBox, VideoToools # Import VideoToools
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Callable, Optional
from .team_identification import TeamIdentification
from torch.utils.tensorboard import SummaryWriter
import collections
import cv2

# DeepSort parameters
MAX_IOU_DISTANCE = 0.7
DEEPSORT_MAX_AGE = 90       # Max frames to keep a track without detection
DEEPSORT_N_INIT = 5         # Min consecutive detections to initialize a track
DEEPSORT_MAX_COSINE_DISTANCE = 0.3 # Max cosine distance for appearance matching

logger = logging.getLogger(__name__)

class VideoModel:
    """Handles the loading and potentially inference of the video processing model."""

    def __init__(self, 
                 model_name: Optional[str] = None,
                 drive_path: Optional[str] = None,
                 writer: Optional[SummaryWriter] = None, 
                 store: Optional[Store] = None, # Add store argument
                 tools: Optional[VideoToools] = None, # Add tools argument
                 device: Optional[torch.device] = None,
                 ):
        """
        Initializes the VideoModel, including the detection and tracking components.

        Args:
            model_name: The name of the model file on Google Drive. Defaults to const.MODEL_NAME.
            drive_path: The path within Google Drive where the model file resides. Defaults to const.GOOGLE_DRIVE_PATH.
            writer: An optional TensorBoard SummaryWriter instance for logging.
            store: An initialized Store object for Google Drive access, passed to DetectionModel.
            tools: An initialized VideoToools instance, passed to DetectionModel for drawing.
            device: The torch.device (cpu or cuda) to load the model onto. Defaults to CPU if None.
        """
        self.model = None  # Model will be loaded later
        self.device = device
        self.writer = writer
        self.tracker = DeepSort(
            max_age=DEEPSORT_MAX_AGE,
            n_init=DEEPSORT_N_INIT,
            max_iou_distance=MAX_IOU_DISTANCE,
            max_cosine_distance=DEEPSORT_MAX_COSINE_DISTANCE 
        )
        # Only pass store if it is not None, otherwise raise an error or provide a default
        if store is None:
            raise ValueError("A valid Store instance must be provided to DetectionModel.")
        self.detection_model = DetectionModel(
            model_name=model_name,  # Pass through, DetectionModel will use its defaults if None
            drive_path=drive_path,  # Pass through, DetectionModel will use its defaults if None
            device=device,
            store=store,  # Pass the store object (guaranteed not None)
            writer=writer,  # Pass the writer
            tools=tools)    # Pass the tools instance
        self.team_identifier = TeamIdentification(writer=self.writer) # Instantiate TeamIdentification
        self.track_to_team_map = {} # For smoothed team assignments
        self.motion_compensator = CameraMotionCompensator()
        self.prev_frame_gray_for_ecc = None # For ECC motion compensation
        self.N_HISTORY_FRAMES = 1  # History window for smoothing
        self.MIN_HISTORY_FOR_CONFIRMATION = 1 # Min observations before a team can be "confirmed"
        self.CONFIRMATION_THRESHOLD_CERTAINTY = 0.0 # Min certainty to change a confirmed team or initially confirm

        # Constants for logging individual track ROIs
        self.LOG_TRACK_IMAGE_FRAME_INTERVAL = 10  # Log track ROIs every Nth call to update_track_team_assignments
        self.MAX_TRACK_ROIS_PER_LOGGED_FRAME = 10    # Max track ROIs to log from such a frame
        self._track_rois_logged_this_call = 0     # Internal counter
    
    def generate_detections(self, frame_in_rgb: np.ndarray, frame_idx: int) -> list:
        """
        Generates detections for the input video frame using the detection model.

        Args:
            frame_in_rgb: The input video frame as a numpy array (RGB).
            frame_idx: The current frame index.
        
        Returns:
            A list of detected objects with their bounding boxes and scores.
        """
        detections = self.detection_model.generate_detections(frame_in_rgb, frame_idx=frame_idx)
        return detections

    def generate_tracks(self, frame_in_rgb: np.ndarray, detections: list) -> list:
        """
        Generates tracks for the detected objects using DeepSort.

        Args:
            frame: The input video frame as a numpy array.
            detections: A list of detected objects with their bounding boxes and scores.
        
        Returns:
            A list of tracked objects with their IDs and bounding boxes.
        """
        
        #tracks = self.tracker.update_tracks(detections, frame=frame_in)
        tracks = self.motion_compensate(frame_in_rgb, detections)
        return tracks
    
    def motion_compensate(self, frame_in_rgb: np.ndarray, detections: list) -> list:
        """
        Generates tracks for the detected objects using DeepSort, with camera motion compensation.
        """
        current_frame_gray = cv2.cvtColor(frame_in_rgb, cv2.COLOR_RGB2GRAY)
        transform_matrix = None # Will hold the transform from prev_frame to current_frame

        if self.prev_frame_gray_for_ecc is not None:
            # Estimate transform from self.prev_frame_gray_for_ecc to current_frame_gray
            # Using a simplified ORB + Homography example here for brevity, ECC is more robust
            # For ECC, you'd call: self.motion_compensator.estimate_transform_ecc(current_frame_gray)
            # This is a placeholder for a robust transform estimation
            try:
                # --- Example using ORB (replace with ECC or your preferred method) ---
                orb = cv2.ORB.create(nfeatures=500) # Limit features for speed
                kp1, des1 = orb.detectAndCompute(self.prev_frame_gray_for_ecc, mask=None)
                kp2, des2 = orb.detectAndCompute(current_frame_gray, mask=None)

                if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    if len(matches) > 10: # Minimum matches for homography
                        src_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
                        dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
                        transform_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # --- End ORB Example ---
            except Exception as e:
                logger.warning(f"Motion estimation failed: {e}")
                transform_matrix = None

        self.prev_frame_gray_for_ecc = current_frame_gray.copy() # Update for next iteration

        compensated_detections = []
        if transform_matrix is not None:
            try:
                # We want to transform current detections to the *previous* frame's coordinate system
                # So we need the inverse transform
                inverse_transform_matrix = np.linalg.inv(transform_matrix)
                for det_bbox_xywh, confidence, class_id in detections:
                    x, y, w, h = det_bbox_xywh
                    # Center point of the current detection
                    center_x, center_y = x + w / 2, y + h / 2
                    
                    # Transform the center point
                    # For homography: [x', y', z']^T = H_inv * [x, y, 1]^T
                    # Then x_transformed = x'/z', y_transformed = y'/z'
                    # For affine: [x', y']^T = A_inv * [x, y, 1]^T (if A_inv is 2x3)
                    
                    # Assuming affine for simplicity here (inverse_transform_matrix is 2x3)
                    # If using homography, the transformation is more complex
                    # This is a simplified transformation of the center point.
                    # A full bbox transformation is more involved.
                    current_center = np.array([[[center_x, center_y]]], dtype=np.float32)
                    if inverse_transform_matrix.shape == (3,3): # Homography
                        transformed_center_homogeneous = inverse_transform_matrix @ np.array([center_x, center_y, 1.0])
                        if transformed_center_homogeneous[2] != 0: # Avoid division by zero
                            transformed_center_x = transformed_center_homogeneous[0] / transformed_center_homogeneous[2]
                            transformed_center_y = transformed_center_homogeneous[1] / transformed_center_homogeneous[2]
                        else:
                            transformed_center_x, transformed_center_y = center_x, center_y # Fallback
                    elif inverse_transform_matrix.shape == (2,3): # Affine
                        transformed_center = cv2.transform(current_center, inverse_transform_matrix)
                        transformed_center_x, transformed_center_y = transformed_center[0][0]
                    else: # Fallback if matrix shape is unexpected
                        transformed_center_x, transformed_center_y = center_x, center_y


                    # Create new bbox in the "stabilized" (previous frame's) coordinate system
                    # Note: Width and height might also need scaling depending on the transform
                    # This simplified version keeps w, h the same, which is only valid for translation/rotation
                    stabilized_x = transformed_center_x - w / 2
                    stabilized_y = transformed_center_y - h / 2
                    compensated_detections.append(
                        ([stabilized_x, stabilized_y, w, h], confidence, class_id)
                    )
                logger.debug(f"Compensated {len(compensated_detections)} detections using estimated transform.")
            except np.linalg.LinAlgError:
                logger.warning("Failed to invert transform matrix. Using original detections.")
                compensated_detections = detections # Fallback
            except Exception as e:
                logger.error(f"Error during detection compensation: {e}")
                compensated_detections = detections # Fallback
        else:
            compensated_detections = detections # No transform, use original detections

        # Determine which frame and which set of detections to use for the tracker
        # The tracker will now operate in the coordinate system of the *previous* frame (more stable)
        # The frame passed to the tracker should correspond to the coordinate system of the detections.
        frame_for_tracker_embedding_rgb = None
        detections_to_pass_to_tracker = []

        if self.prev_frame_gray_for_ecc is not None and transform_matrix is not None: # Only use prev frame if compensation happened
            frame_for_tracker_embedding_rgb = cv2.cvtColor(self.prev_frame_gray_for_ecc, cv2.COLOR_GRAY2RGB)
            detections_to_pass_to_tracker = compensated_detections # Use compensated detections
            # frame_height, frame_width = frame_for_tracker_embedding_rgb.shape[:2] # Moved down
        else: # First frame or no transform, use current frame and original detections
            frame_for_tracker_embedding_rgb = frame_in_rgb
            detections_to_pass_to_tracker = detections # Use original detections
            # frame_height, frame_width = frame_for_tracker_embedding_rgb.shape[:2] # Moved down

        # Get dimensions from the actual frame that will be used for cropping by the tracker
        frame_height, frame_width = frame_for_tracker_embedding_rgb.shape[:2]

        # --- Clip bounding boxes to frame boundaries before passing to tracker ---
        clipped_detections = []
        for det_bbox_xywh, confidence, class_id in detections_to_pass_to_tracker:
            x, y, w, h = det_bbox_xywh
            
            # Convert to xyxy for easier clipping
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Clip coordinates to frame boundaries
            x1_clipped = max(0, round(x1)) # Use round for float coords then int
            y1_clipped = max(0, round(y1))
            x2_clipped = min(frame_width, round(x2)) # Ensure x2_clipped does not exceed frame_width
            y2_clipped = min(frame_height, round(y2))# Ensure y2_clipped does not exceed frame_height

            # Re-calculate width and height after clipping
            w_clipped = x2_clipped - x1_clipped
            h_clipped = y2_clipped - y1_clipped

            # Only add the detection if it still has positive dimensions after clipping
            if w_clipped > 0 and h_clipped > 0:
                clipped_detections.append(
                    # Pass xywh to tracker, ensuring x1_clipped, y1_clipped are the top-left
                    ([x1_clipped, y1_clipped, w_clipped, h_clipped], 
                     confidence, 
                     class_id)
                )
            else:
                logger.debug(f"Skipping detection after clipping resulted in zero or negative dimensions: original xywh=({x},{y},{w},{h}), clipped xywh=({x1_clipped},{y1_clipped},{w_clipped},{h_clipped})")

        # Now pass the clipped detections to the tracker
        tracks = self.tracker.update_tracks(clipped_detections, frame=frame_for_tracker_embedding_rgb)

        # The output tracks from self.tracker.update_tracks will have bbox coordinates
        # in the "stabilized" (previous frame's) coordinate system.
        # We need to transform them *back* to the current frame's coordinate system for drawing.
        output_tracks = []
        if transform_matrix is not None:
            for track in tracks:
                if track.is_confirmed(): # Process only confirmed tracks for transformation
                    # Assuming track.to_tlwh() gives [x,y,w,h] in the stabilized system
                    x_s, y_s, w_s, h_s = track.to_tlwh() 
                    center_x_s, center_y_s = x_s + w_s / 2, y_s + h_s / 2

                    # Transform center point back to current frame coordinates
                    # This is a simplified transformation of the center point.
                    stabilized_center = np.array([[[center_x_s, center_y_s]]], dtype=np.float32)
                    if transform_matrix.shape == (3,3): # Homography
                        current_center_homogeneous = transform_matrix @ np.array([center_x_s, center_y_s, 1.0])
                        if current_center_homogeneous[2] != 0:
                            current_center_x = current_center_homogeneous[0] / current_center_homogeneous[2]
                            current_center_y = current_center_homogeneous[1] / current_center_homogeneous[2]
                        else:
                            current_center_x, current_center_y = center_x_s, center_y_s # Fallback
                    elif transform_matrix.shape == (2,3): # Affine
                        current_center = cv2.transform(stabilized_center, transform_matrix)
                        current_center_x, current_center_y = current_center[0][0]
                    else: # Fallback
                        current_center_x, current_center_y = center_x_s, center_y_s

                    # Create a new representation of the track or update its bbox for the current frame
                    # For simplicity, we'll update a copy of the track's ltwh if possible,
                    # or store the transformed bbox alongside the track.
                    # Directly modifying track.ltwh might not be safe depending on DeepSort's internals.
                    # A safer way is to pass the transform_matrix to the drawing function.
                    # However, to make tracks usable by downstream logic expecting current frame coords:
                    current_x = current_center_x - w_s / 2
                    current_y = current_center_y - h_s / 2
                    
                    # Re-assigning to track.ltwh might not be the intended use of the Track object.
                    # It's better to store this transformed bbox separately or adjust drawing.
                    # For now, we'll append the original track, assuming drawing will handle transformation.
                    output_tracks.append(track) # Track's bbox is still in stabilized coords.
        else:
            output_tracks = tracks

        return output_tracks # These tracks might be in stabilized coords if transform_matrix was used

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
            original_detection_bbox = BoundingBox(x1=original_ltwh[0],
                                                  y1=original_ltwh[1],
                                                  w=original_ltwh[2],
                                                  h=original_ltwh[3])
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

class CameraMotionCompensator:
    def __init__(self):
        self.prev_gray = None
        # For ECC
        self.warp_mode = cv2.MOTION_HOMOGRAPHY # cv2.MOTION_AFFINE cv2.MOTION_EUCLIDEAN
        self.number_of_iterations = 50
        self.termination_eps = 1e-8
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         self.number_of_iterations,
                         self.termination_eps)

    def estimate_transform_ecc(self, current_frame_gray: np.ndarray) -> Optional[np.ndarray]:
        if self.prev_gray is None:
            self.prev_gray = current_frame_gray
            return np.eye(2, 3, dtype=np.float32) if self.warp_mode != cv2.MOTION_HOMOGRAPHY else np.eye(3, 3, dtype=np.float32)

        try:
            warp_matrix = np.zeros((3, 3), dtype=np.float32) if self.warp_mode == cv2.MOTION_HOMOGRAPHY else np.zeros((2, 3), dtype=np.float32)
            np.fill_diagonal(warp_matrix, 1)
            cc, warp_matrix = cv2.findTransformECC(self.prev_gray, current_frame_gray, warp_matrix,
                                                   self.warp_mode, self.criteria)
            self.prev_gray = current_frame_gray
            return warp_matrix
        except cv2.error as e:
            logger.warning(f"ECC transform estimation failed: {e}. Using identity matrix.")
            self.prev_gray = current_frame_gray # Update prev_gray anyway
            return np.eye(2, 3, dtype=np.float32) if self.warp_mode != cv2.MOTION_HOMOGRAPHY else np.eye(3, 3, dtype=np.float32)


# In VideoModel.__init__
# self.motion_compensator = CameraMotionCompensator()

# In VideoModel.generate_tracks (or a new method called before it)
# current_frame_gray = cv2.cvtColor(frame_in, cv2.COLOR_RGB2GRAY)
# transform_matrix = self.motion_compensator.estimate_transform_ecc(current_frame_gray)
#
# Now, how to use transform_matrix with DeepSort:
# DeepSort itself doesn't directly take this matrix. You'd need to:
# 1. Modify DeepSort's Kalman Filter: This is complex. The Kalman filter's motion model
#    would need to incorporate this external camera motion.
# 2. Adjust Detections or Track States *before* `tracker.update_tracks()`:
#    - If you adjust detections: For each detection bbox (x,y,w,h), transform its center (x+w/2, y+h/2)
#      using the inverse of `transform_matrix` (to bring it to the "stabilized" previous frame's coordinates).
#      Then feed these transformed detections to DeepSort. This is often the more practical approach.
#      The output tracks will be in the "stabilized" coordinate system, so you'd need to transform them
#      back using `transform_matrix` for drawing on the current `frame_in`.

   