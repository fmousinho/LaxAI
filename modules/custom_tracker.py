import logging
from typing import Optional, List, Tuple
import numpy as np
import cv2
import supervision as sv
from .Siglip_reid import SiglipReID


logger = logging.getLogger(__name__)
# Set the logging level specifically for this module to INFO.
# This will prevent DEBUG messages from this module from being displayed.

_TRACK_ACTIVATION_THRESHOLD = 0.2
_LOST_TRACK_BUFFER = 15
_MINIMUM_MATCHING_THRESHOLD = 0.7
_MINIMUM_CONSECUTIVE_FRAMES = 30    

def warp_bbox(bbox_tlbr: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    """
    Applies an affine transformation to a single bounding box.

    The input bounding box is expected in [x_min, y_min, x_max, y_max] (tlbr) format.
    The affine matrix should be a 2x3 transformation matrix.
    The function transforms the four corners of the bounding box and then
    computes the new axis-aligned bounding box that encloses the warped corners.

    Args:
        bbox_tlbr (np.ndarray): A 1D NumPy array representing the bounding box
            coordinates as [x_min, y_min, x_max, y_max].
        affine_matrix (np.ndarray): A 2x3 NumPy array representing the affine
            transformation matrix.

    Returns:
        np.ndarray: A 1D NumPy array representing the warped bounding box
            coordinates as [x_min, y_min, x_max, y_max].
    """
    # Ensure input bounding box coordinates are float32 for all calculations within this function.
    if bbox_tlbr.dtype != np.float32:
        bbox_tlbr = bbox_tlbr.astype(np.float32)

    # The affine matrix from OpenCV is 2x3. We need a 3x3 for matrix multiplication
    # on homogenous coordinates.
    m_3x3 = np.vstack([affine_matrix, [0, 0, 1]])

    # Get the four corners of each bounding box
    x1 = bbox_tlbr[0]
    y1 = bbox_tlbr[1]
    x2 = bbox_tlbr[2]
    y2 = bbox_tlbr[3]

    # Create an array of 4 corners for the box: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    # The current reshape(-1,4,2) implies it's prepared for a batch of 1.
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    # Convert to homogenous coordinates by adding a '1'
    # corners_hom will be (4, 3)
    corners_hom = np.concatenate(
        [corners, np.ones((4, 1))], axis=1
    )

    # Apply the transformation matrix
    # m_3x3 is (3,3), corners_hom.T is (3,4)
    # warped_corners_hom_transposed is (3,4)
    warped_corners_hom_transposed = m_3x3 @ corners_hom.T
    # warped_corners_hom is (4,3)
    warped_corners_hom = warped_corners_hom_transposed.T

    # Convert back from homogenous coordinates
    # Ensure division is done correctly for each point
    warped_corners = warped_corners_hom[:, :2] / warped_corners_hom[:, 2, np.newaxis]


    # Re-calculate the new axis-aligned bounding boxes (xyxy)
    min_xy = warped_corners.min(axis=0) # min_xy will be (2,)
    max_xy = warped_corners.max(axis=0) # max_xy will be (2,)

    warped_xyxy = np.hstack([np.maximum(0, min_xy), np.maximum(0, max_xy)]) # Ensure no negative coordinates

    return warped_xyxy

class AffineAwareByteTrack(sv.ByteTrack):
    """
    A custom ByteTrack tracker that incorporates affine motion compensation and
    reidentification using siglip embeddings (external module).

    This class extends `supervision.ByteTrack` to adjust the predicted
    locations of tracks based on an estimated affine transformation between
    consecutive frames. This helps in maintaining track identity even when
    there is camera motion.
    """

    def __init__(self, frame_rate: int, *args, **kwargs):
        super().__init__(
            frame_rate = frame_rate,
            track_activation_threshold = _TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer = _LOST_TRACK_BUFFER,
            minimum_matching_threshold = _MINIMUM_MATCHING_THRESHOLD,
            minimum_consecutive_frames = _MINIMUM_CONSECUTIVE_FRAMES,
            *args, **kwargs)
        self.reid = SiglipReID()
        self.reassigned_track_ids: set[int] = set()  # Track IDs that have been reassigned in the current frame

    def update_with_transform(self, detections: sv.Detections, affine_matrix: np.ndarray, frame: np.ndarray) -> sv.Detections:
        """_summary_
        Updates the tracker with new detections after applying affine compensation.

        Before calling the standard `update_with_detections` method of the parent
        `ByteTrack` class, this method first transforms the predicted bounding
        boxes of all existing tracks (tracked, lost, removed) using the provided
        affine matrix. This aligns the track predictions with the current frame's
        coordinate system, improving matching with new detections.

        Args:
            detections (sv.Detections): The new detections for the current frame.
            affine_matrix (np.ndarray): A 2x3 affine transformation matrix that
            frame (np.ndarray): The raw image of the current frame, needed for
                maps coordinates from the previous frame to the current frame.

        Returns:
            sv.Detections: The detections object, potentially with updated tracker IDs
                and other tracking information, after processing by the ByteTrack algorithm.
        """
        # Wwarp the internal tracks' predicted locations
        # ByteTrack keeps the following 3 types of lists of tracks, all in the STtrack format
        for track_list_type in [self.tracked_tracks, self.lost_tracks]:
            for track in track_list_type:
                if track.mean is None or track.covariance is None: # Should not happen for active/lost tracks
                    continue

                current_bbox_tlbr = track.tlbr # Derived from track.mean, shape (4,)
                warped_bbox_tlbr = warp_bbox(current_bbox_tlbr, affine_matrix) # Expected shape (4,)
                
                # Convert warped_bbox_tlbr to [cx, cy, a, h] for Kalman state
                w_x1, w_y1, w_x2, w_y2 = warped_bbox_tlbr
                w_w = w_x2 - w_x1
                w_h = w_y2 - w_y1

                if w_w <= 0 or w_h <= 0: # Warped box is degenerate, skip update for this track
                    logger.debug(f"Track {track.internal_track_id} warped to degenerate bbox: {warped_bbox_tlbr}. Skipping state update.")
                    continue

                new_cx = w_x1 + w_w / np.float32(2.0)
                new_cy = w_y1 + w_h / np.float32(2.0)
                new_a = w_w / w_h 
                new_h = w_h

                # Update positional components of track.mean
                # track.mean is [cx, cy, a, h, vcx, vcy, va, vh]
                track.mean[0] = new_cx # new_cx is already np.float32 due to earlier casting
                track.mean[1] = new_cy # new_cy is already np.float32
                track.mean[2] = new_a  # new_a is already np.float32
                track.mean[3] = new_h  # new_h is already np.float32

                # Reset the entire covariance matrix to reflect the new state's uncertainty,
                # similar to KalmanFilter.initiate().
                # We do NOT reset track.mean[4:8] (velocities); the filter will adjust them.
                
                # pylint: disable=protected-access 
                std_pos_w = track.kalman_filter._std_weight_position
                std_vel_w = track.kalman_filter._std_weight_velocity
                # pylint: enable=protected-access

                current_h = track.mean[3] # This is new_h

                std = [
                    2 * std_pos_w * current_h,       # std_cx
                    2 * std_pos_w * current_h,       # std_cy
                    1e-2,                            # std_a (aspect ratio std is usually small and fixed)
                    2 * std_pos_w * current_h,       # std_h
                    10 * std_vel_w * current_h,      # std_vcx
                    10 * std_vel_w * current_h,      # std_vcy
                    1e-1,                            # std_va (aspect ratio velocity std, 1e-5 in supervision is very small, 1e-1 or 1e-2 is more common)
                    10 * std_vel_w * current_h       # std_vh
                ]
                
                # Ensure all std values are at least a small epsilon to avoid issues
                # if current_h is zero or extremely small (though earlier checks should prevent zero current_h)
                min_std_val = 1e-6
                std_np = np.maximum(np.array(std, dtype=np.float32), min_std_val)

                track.covariance = np.diag(np.square(std_np))

        logger.debug("Applied affine transform to %d tracked, %d lost, %d removed tracks before update.",
                     len(self.tracked_tracks), len(self.lost_tracks), len(self.removed_tracks))
        
        detections = super().update_with_detections(detections)

        # Create a map from external_track_id to internal_track_id for currently tracked tracks.
        external_to_internal_id_map = {
            track.external_track_id: track.internal_track_id
            for track in self.tracked_tracks
            if track.external_track_id != -1
        }

        # Replace external track IDs in detections with their internal counterparts.
        if detections.tracker_id is not None:
            # Using .get() provides a default value (-1) for IDs not in the map.
            internal_ids = np.array([
                external_to_internal_id_map.get(int(ext_id), -1)
                for ext_id in detections.tracker_id
            ], dtype=int)
            detections.tracker_id = internal_ids

        self.reid.update_ids(detections=detections, frame=frame, frame_id=self.frame_id)

        return detections
    
    @staticmethod
    def calculate_affine_transform(prev_frame: np.ndarray, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates the affine transformation matrix between two frames.

        This method uses feature matching (Good Features to Track and Lucas-Kanade
        optical flow) to estimate the 2D affine transformation that maps points
        from the `prev_frame` to the `current_frame`.

        Args:
            prev_frame (np.ndarray): The previous video frame (BGR format).
            current_frame (np.ndarray): The current video frame (BGR format).

        Returns:
            Optional[np.ndarray]: A 2x3 affine transformation matrix if successful,
            otherwise None (e.g., if not enough points are found for estimation).
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Find good features to track
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

        if prev_pts is None:
            logger.warning("Could not calculate affine matrix because prev_pts is None (no good features found in previous frame).")
            return None

        # Calculate optical flow
        current_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_pts, None) 

        # Filter out bad points
        if current_pts is not None and status is not None:
            good_new = current_pts[status == 1]
            good_old = prev_pts[status == 1]
        else:
            logger.warning("Could not calculate affine matrix because current_pts or status is None after optical flow.")
            return None


        # We need at least 3 points to estimate the affine transform
        if len(good_new) < 3:
            logger.warning("Could not calculate affine matrix: not enough good points found after optical flow (found %d, need at least 3).", len(good_new))
            return None

        # Estimate the affine transformation matrix
        # This matrix will map points from the prev_gray frame to the current_gray frame
        m, mask = cv2.estimateAffine2D(good_old, good_new)
        
        if m is None:
            logger.warning("cv2.estimateAffine2D returned None, failed to estimate affine transform.")
            return None

        # Ensure the affine matrix is float32
        m = m.astype(np.float32)

        logger.debug("Successfully calculated affine transform matrix.")
        return m

    @staticmethod
    def get_identity_affine_matrix() -> np.ndarray:
        """
        Returns a 2x3 identity affine transformation matrix.

        This matrix represents no transformation (scaling, rotation, or translation).
        It can be used as a default when an affine transform
        cannot be calculated or is not needed.

        Returns:
            np.ndarray: A 2x3 float32 NumPy array representing the identity
            affine matrix.
        """
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)

    def get_reid_data_for_tid(self, tid: int) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Retrieves re-identification data (embeddings and crops) for a given tracker ID.

        This method serves as a public interface to the internal SiglipReID module,
        allowing the application to get ReID data without directly interacting with
        the SiglipReID class.

        Args:
            tid (int): The tracker ID.

        Returns:
            Optional[Tuple[np.ndarray, List[np.ndarray]]]: A tuple of (embedding, crops) or None.
        """
        return self.reid.get_embeddings_and_crops_by_tid(tid)
    
    def orphan_track_ids(self) -> List[int]:
        """
        Returns a list of tracker IDs that were considered 'removed' by ByteTrack
        in the latest update.

        These tracks are candidates for re-identification.
        Returns:
            List[int]: A list of tracker IDs for tracks newly marked as 'removed'.
        """
        # self.removed_tracks is a list of STrack objects. Accessing an attribute
        # directly on the list will fail. We must iterate through the list using
        # a list comprehension to extract the ID from each track.
        
        byte_track_removed_ids = set()
        for track in self.removed_tracks:
            # We return the internal_track_id for tracks that were once active (had an external_track_id).
            # This is consistent with the rest of the system now using internal IDs.
            byte_track_removed_ids.add(track.internal_track_id)
        
        result = byte_track_removed_ids - self.reassigned_track_ids
        result = list(result)

        logger.debug("Orphan track IDs: %s", result)

        return result
    
