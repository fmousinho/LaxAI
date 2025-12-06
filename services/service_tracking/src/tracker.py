import logging
logger = logging.getLogger(__name__)

from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import supervision as sv
import torch

from shared_libs.config.all_config import model_config, tracker_config




def warp_bbox(
    bbox_tlbr: Union[np.ndarray, "torch.Tensor"], affine_matrix: Union[np.ndarray, "torch.Tensor"]
) -> np.ndarray:
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
    # Auto-dispatch to torch implementation when tensors are provided.
    # Accept either a single bbox (shape (4,)) or a batch (N,4) when using torch.
    try:
        is_torch = isinstance(bbox_tlbr, torch.Tensor) or isinstance(affine_matrix, torch.Tensor)
    except Exception:
        is_torch = False

    if is_torch:
        # Ensure bbox is a torch tensor of shape (N,4)
        if not isinstance(bbox_tlbr, torch.Tensor):
            bbox_t = torch.as_tensor(bbox_tlbr, dtype=torch.float32)
        else:
            bbox_t = bbox_tlbr

        # If single bbox provided as 1-D, make it batch of 1
        if bbox_t.dim() == 1:
            bbox_t = bbox_t.unsqueeze(0)

        # Ensure affine is tensor
        if not isinstance(affine_matrix, torch.Tensor):
            affine_t = torch.as_tensor(affine_matrix, dtype=torch.float32, device=bbox_t.device)
        else:
            affine_t = affine_matrix.to(bbox_t.device)

        # Call the batched torch implementation
        warped = warp_bbox_torch(bbox_t, affine_t)

        # Return the first (and only) warped bbox as numpy array if original input was 1-D
        out = warped.cpu().numpy()
        if out.shape[0] == 1:
            return out[0]
        return out

    # Numpy fallback (existing behavior)
    # Convert inputs to NumPy arrays (this handles cases where inputs are array-like)
    bbox_tlbr = np.asarray(bbox_tlbr, dtype=np.float32)
    affine_matrix = np.asarray(affine_matrix, dtype=np.float32)

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
    corners_hom = np.concatenate([corners, np.ones((4, 1))], axis=1)

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
    min_xy = warped_corners.min(axis=0)  # min_xy will be (2,)
    max_xy = warped_corners.max(axis=0)  # max_xy will be (2,)

    warped_xyxy = np.hstack(
        [np.maximum(0, min_xy), np.maximum(0, max_xy)]
    )  # Ensure no negative coordinates

    return warped_xyxy


def warp_bbox_torch(bboxes: "torch.Tensor", affine: "torch.Tensor") -> "torch.Tensor":
    """
    Batched bbox warping using torch tensors. Accepts bboxes of shape (N,4)
    and affine matrices of shape (N,2,3) or (2,3) (broadcastable).

    Returns a tensor of shape (N,4) with clipped coordinates (>=0).
    """
    # Ensure float32
    bboxes = bboxes.float()
    device = bboxes.device

    N = bboxes.shape[0]

    # corners: (N,4,2)
    corners = torch.stack(
        [
            torch.stack([bboxes[:, 0], bboxes[:, 1]], dim=1),
            torch.stack([bboxes[:, 2], bboxes[:, 1]], dim=1),
            torch.stack([bboxes[:, 2], bboxes[:, 3]], dim=1),
            torch.stack([bboxes[:, 0], bboxes[:, 3]], dim=1),
        ],
        dim=1,
    ).to(device)

    ones = torch.ones((N, 4, 1), device=device, dtype=torch.float32)
    corners_h = torch.cat([corners, ones], dim=2)  # (N,4,3)

    # Prepare affine matrices to (N,3,3)
    if affine.dim() == 2:
        affine = affine.unsqueeze(0).expand(N, -1, -1)
    # Build 3x3 matrices
    bottom = (
        torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        .view(1, 1, 3)
        .expand(N, 1, 3)
    )
    M = torch.cat([affine, bottom], dim=1)  # (N,3,3)

    # Apply transformation: (N,3,3) @ (N,3,4) -> (N,3,4)
    warped = (M @ corners_h.permute(0, 2, 1)).permute(0, 2, 1)  # (N,4,3)
    warped_xy = warped[:, :, :2] / warped[:, :, 2:].clamp(min=1e-6)

    min_xy, _ = warped_xy.min(dim=1)
    max_xy, _ = warped_xy.max(dim=1)

    warped_xyxy = torch.cat([torch.clamp(min_xy, min=0.0), torch.clamp(max_xy, min=0.0)], dim=1)
    return warped_xyxy


class TrackData:
    """
    A simple data class to hold re-identification data for a track.

    This class stores crop images, embeddings, and metadata for individual
    tracks across multiple frames to support re-identification and team
    classification tasks.
    """

    def __init__(
        self,
        track_id: int,
        crop: np.ndarray,
        class_id: int,
        confidence: float,
        frame_id: int,
    ) -> None:
        """
        Initialize track data for a new track.

        Args:
            track_id: Unique identifier for the track
            crop: Initial crop image for the track
            class_id: Class identifier for the detected object
            confidence: Confidence score for the detection
            frame_id: Frame number where track was first detected
        """
        self.track_id = track_id
        self.crops: list[np.ndarray] = [crop]
        self._embedding: np.ndarray = np.empty((0,))
        self._class_id = class_id
        self._class_confidence = confidence
        self.frame_first_seen: int = frame_id
        self.frame_last_seen: int = frame_id
        self._team: int = -1

    @property
    def embedding(self) -> np.ndarray:
        """Get the embedding vector for this track."""
        return self._embedding

    @embedding.setter
    def embedding(self, value: np.ndarray) -> None:
        """Set the embedding vector for this track."""
        self._embedding = value

    @property
    def team(self) -> int:
        """Get the team identifier for this track."""
        return self._team

    @team.setter
    def team(self, value: int) -> None:
        """Set the team identifier for this track."""
        self._team = value

    @property
    def class_id(self) -> int:
        """Get the class identifier for this track."""
        return self._class_id

    @property
    def num_crops(self) -> int:
        """Return the number of crops stored for this track."""
        return len(self.crops)

    def update_data(self, crop: np.ndarray, class_id: int, confidence: float) -> None:
        """
        Update the track data with a new crop, class ID, and confidence.

        Args:
            crop: New crop image to add
            class_id: Class identifier for the detection
            confidence: Confidence score for the detection
        """
        if confidence > self._class_confidence:
            self._class_confidence = confidence
            self._class_id = class_id
        self.crops.append(crop)
        self.frame_last_seen += 1

    def update_metadata(self, class_id: int, confidence: float) -> None:
        """
        Update the track metadata without adding a crop.

        Args:
            class_id: Class identifier for the detection
            confidence: Confidence score for the detection
        """
        if confidence > self._class_confidence:
            self._class_confidence = confidence
            self._class_id = class_id
        self.frame_last_seen += 1


class AffineAwareByteTrack(sv.ByteTrack):
    """
    A custom ByteTrack tracker that incorporates affine motion compensation and
    reidentification using siglip embeddings (external module).

    This class extends `supervision.ByteTrack` to adjust the predicted
    locations of tracks based on an estimated affine transformation between
    consecutive frames. This helps in maintaining track identity even when
    there is camera motion.
    """

    def __init__(
        self,
        maintain_separate_track_obj: bool = True,
        crop_save_interval: int = 5,
    ) -> None:
        """
        Initialize the AffineAwareByteTrack tracker.

        Args:
            maintain_separate_track_obj: Whether to maintain separate track objects
            crop_save_interval: Interval for saving crops
        """
        super().__init__(
            track_activation_threshold=tracker_config.track_activation_threshold,
            lost_track_buffer=tracker_config.lost_track_buffer,
            minimum_matching_threshold=tracker_config.minimum_matching_threshold,
            frame_rate=30,
            minimum_consecutive_frames=tracker_config.minimum_consecutive_frames,
        )
        self.track_data: dict[int, TrackData] = {}
        self.id_type = tracker_config.id_type
        self.maintain_separate_track_obj = maintain_separate_track_obj
        self.crop_save_interval = crop_save_interval
        self.frame_count = 0

    def update_with_transform(
        self, detections: sv.Detections, affine_matrix: np.ndarray, frame: np.ndarray
    ) -> sv.Detections:
        """
        Update the tracker with new detections after applying affine compensation.

        Before calling the standard `update_with_detections` method of the parent
        `ByteTrack` class, this method first transforms the predicted bounding
        boxes of all existing tracks (tracked, lost, removed) using the provided
        affine matrix. This aligns the track predictions with the current frame's
        coordinate system, improving matching with new detections.

        Args:
            detections: The new detections for the current frame
            affine_matrix: A 2x3 affine transformation matrix that maps coordinates
                from the previous frame to the current frame
            frame: The raw image of the current frame, needed for extracting crops

        Returns:
            The detections object, potentially with updated tracker IDs and other
            tracking information, after processing by the ByteTrack algorithm.
        """
        if len(detections) == 0:
            self.frame_count += 1
            return detections

        # Warp the internal tracks' predicted locations
        # ByteTrack keeps the following 3 types of lists of tracks, all in the STtrack format
        for track_list_type in [self.tracked_tracks, self.lost_tracks]:
            for track in track_list_type:
                if (
                    track.mean is None or track.covariance is None
                ):  # Should not happen for active/lost tracks
                    continue

                current_bbox_tlbr = track.tlbr  # Derived from track.mean, shape (4,)
                warped_bbox_tlbr = warp_bbox(
                    current_bbox_tlbr, affine_matrix
                )  # Expected shape (4,)

                # Convert warped_bbox_tlbr to [cx, cy, a, h] for Kalman state
                w_x1, w_y1, w_x2, w_y2 = warped_bbox_tlbr
                w_w = w_x2 - w_x1
                w_h = w_y2 - w_y1

                # Convert tensor values to scalars to avoid TracerWarning with boolean conversion
                w_w_val = w_w.item() if isinstance(w_w, torch.Tensor) else w_w
                w_h_val = w_h.item() if isinstance(w_h, torch.Tensor) else w_h
                
                if w_w_val <= 0 or w_h_val <= 0:  # Warped box is degenerate, skip update for this track
                    logger.debug(
                        f"Track {track.internal_track_id} warped to degenerate bbox: {warped_bbox_tlbr}. Skipping state update."
                    )
                    continue

                # Convert tensor values to scalars for consistent arithmetic operations
                w_x1_val = w_x1.item() if isinstance(w_x1, torch.Tensor) else w_x1
                w_y1_val = w_y1.item() if isinstance(w_y1, torch.Tensor) else w_y1
                w_w_val = w_w.item() if isinstance(w_w, torch.Tensor) else w_w
                w_h_val = w_h.item() if isinstance(w_h, torch.Tensor) else w_h
                
                new_cx = w_x1_val + w_w_val / 2.0
                new_cy = w_y1_val + w_h_val / 2.0
                new_a = w_w_val / w_h_val
                new_h = w_h_val

                # Update positional components of track.mean
                # track.mean is [cx, cy, a, h, vcx, vcy, va, vh]
                track.mean[0] = new_cx
                track.mean[1] = new_cy
                track.mean[2] = new_a
                track.mean[3] = new_h

                # Extract 2x2 affine matrix for velocity transformations
                affine_2x2 = affine_matrix[:2, :2]

                # Transform velocities based on configuration
                if tracker_config.transform_velocities:
                    velocity_xy = np.array([track.mean[4], track.mean[5]])
                    transformed_velocity_xy = affine_2x2 @ velocity_xy

                    track.mean[4] = transformed_velocity_xy[0]  # transformed vcx
                    track.mean[5] = transformed_velocity_xy[1]  # transformed vcy

                # Optionally scale height velocity if there's significant scaling
                if tracker_config.scale_height_velocity:
                    scale_factor = np.sqrt(np.linalg.det(affine_2x2))
                    if abs(scale_factor - 1.0) > tracker_config.scaling_threshold:
                        track.mean[7] *= scale_factor

                # Reset the entire covariance matrix to reflect the new state's uncertainty,
                # similar to KalmanFilter.initiate().
                # We do NOT reset track.mean[4:8] (velocities); the filter will adjust them.

                # pylint: disable=protected-access
                std_pos_w = self.kalman_filter._std_weight_position
                std_vel_w = self.kalman_filter._std_weight_velocity
                # pylint: enable=protected-access

                current_h = track.mean[3]  # This is new_h

                std = [
                    2 * std_pos_w * current_h,  # std_cx
                    2 * std_pos_w * current_h,  # std_cy
                    1e-2,  # std_a (aspect ratio std is usually small and fixed)
                    2 * std_pos_w * current_h,  # std_h
                    10 * std_vel_w * current_h,  # std_vcx
                    10 * std_vel_w * current_h,  # std_vcy
                    1e-1,  # std_va (aspect ratio velocity std, 1e-5 in supervision is very small, 1e-1 or 1e-2 is more common)
                    10 * std_vel_w * current_h,  # std_vh
                ]

                # Ensure all std values are at least a small epsilon to avoid issues
                # if current_h is zero or extremely small (though earlier checks should prevent zero current_h)
                min_std_val = 1e-6
                std_np = np.maximum(np.array(std, dtype=np.float32), min_std_val)

                track.covariance = np.diag(np.square(std_np))

        logger.debug(
            "Applied affine transform to %d tracked, %d lost, %d removed tracks before update.",
            len(self.tracked_tracks),
            len(self.lost_tracks),
            len(self.removed_tracks),
        )

        detections = super().update_with_detections(detections)

        if self.id_type == "internal":
            # Create a map from external_track_id to internal_track_id for currently tracked tracks.
            external_to_internal_id_map = {
                track.external_track_id: track.internal_track_id
                for track in self.tracked_tracks
                if track.external_track_id != -1
            }
            # Replace external track IDs in detections with their internal counterparts.
            if detections.tracker_id is not None:
                # Using .get() provides a default value (-1) for IDs not in the map.
                internal_ids = np.array(
                    [
                        external_to_internal_id_map.get(int(ext_id), -1)
                        for ext_id in detections.tracker_id
                    ],
                    dtype=int,
                )
                detections.tracker_id = internal_ids

        # Updates crops for each track
        if self.maintain_separate_track_obj:
            self._update_tracks_obj(detections, frame)

        # Increment frame count for crop saving logic
        self.frame_count += 1

        return detections

    @staticmethod
    def calculate_affine_transform(
        prev_frame: np.ndarray, current_frame: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the affine transformation matrix between two frames.

        This method uses feature matching (Good Features to Track and Lucas-Kanade
        optical flow) to estimate the 2D affine transformation that maps points
        from the `prev_frame` to the `current_frame`.

        Args:
            prev_frame: The previous video frame (BGR format)
            current_frame: The current video frame (BGR format)

        Returns:
            A 2x3 affine transformation matrix if successful, otherwise returns an identity.
            (e.g., if not enough points are found for estimation).
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Find good features to track
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30
        )

        if prev_pts is None:
            logger.warning(
                "Could not calculate affine matrix because prev_pts is None (no good features found in previous frame)."
            )
            return AffineAwareByteTrack.get_identity_affine_matrix()

        # Calculate optical flow
        current_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, current_gray, prev_pts, None  # type: ignore
        )

        # Filter out bad points
        if current_pts is not None and status is not None:
            good_new = current_pts[status == 1]
            good_old = prev_pts[status == 1]
        else:
            logger.warning(
                "Could not calculate affine matrix because current_pts or status is None after optical flow."
            )
            return AffineAwareByteTrack.get_identity_affine_matrix()

        # We need at least 3 points to estimate the affine transform
        if len(good_new) < 3:
            logger.warning(
                "Could not calculate affine matrix: not enough good points found after optical flow (found %d, need at least 3).",
                len(good_new),
            )
            return AffineAwareByteTrack.get_identity_affine_matrix()

        # Vectorized displacement analysis to detect near-static scenes
        displacement = good_new - good_old
        # Norms of displacement for all tracked points (per-point translation magnitude)
        displacement_norms = np.linalg.norm(displacement, axis=1)

        if displacement_norms.size == 0:
            logger.debug("Optical flow returned zero displacement points; treating as static camera.")
            return AffineAwareByteTrack.get_identity_affine_matrix()

        median_disp = float(np.median(displacement_norms))
        max_disp = float(np.max(displacement_norms))

        if (
            median_disp <= tracker_config.affine_median_translation_px
            and max_disp <= tracker_config.affine_max_translation_px
        ):
            logger.debug(
                "Skipping affine warp (median displacement=%.3fpx, max=%.3fpx)",
                median_disp,
                max_disp,
            )
            return AffineAwareByteTrack.get_identity_affine_matrix()

        # Estimate the affine transformation matrix
        # This matrix will map points from the prev_gray frame to the current_gray frame
        m, mask = cv2.estimateAffine2D(good_old, good_new)

        if m is None:
            logger.warning(
                "cv2.estimateAffine2D returned None, failed to estimate affine transform."
            )
            return AffineAwareByteTrack.get_identity_affine_matrix()

        # Ensure the affine matrix is float32
        m = m.astype(np.float32)

        # If the resulting transform is approximately identity, skip warping to reduce noise
        translation_mag = float(np.linalg.norm(m[:, 2]))
        linear_delta = float(np.linalg.norm(m[:, :2] - np.eye(2, dtype=np.float32)))

        if (
            translation_mag <= tracker_config.affine_median_translation_px
            and linear_delta <= tracker_config.affine_linear_fro_threshold
        ):
            logger.debug(
                "Affine matrix ~ identity (translation=%.3fpx, linear Δ=%.4f); using identity.",
                translation_mag,
                linear_delta,
            )
            return AffineAwareByteTrack.get_identity_affine_matrix()

        logger.debug(
            "Calculated affine transform (translation=%.3fpx, linear Δ=%.4f).",
            translation_mag,
            linear_delta,
        )
        return m

    @staticmethod
    def get_identity_affine_matrix() -> np.ndarray:
        """
        Return a 2x3 identity affine transformation matrix.

        This matrix represents no transformation (scaling, rotation, or translation).
        It can be used as a default when an affine transform
        cannot be calculated or is not needed.

        Returns:
            A 2x3 float32 NumPy array representing the identity affine matrix.
        """
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def get_tids_for_frame(self) -> List[int]:
        """
        Retrieve all unique track IDs for the current frame, including lost tracks.

        Returns:
            A sorted list of unique track IDs for the specified frame.
        """
        tids = set()

        # Collect external track IDs from tracked tracks and lost tracks
        tids.update(track.internal_track_id for track in self.tracked_tracks)
        tids.update(track.internal_track_id for track in self.lost_tracks)

        return sorted(list(tids))

    def _update_tracks_obj(self, detections: sv.Detections, frame: np.ndarray) -> None:
        """
        Update the track data with the latest detections and crops.
        Only saves crops every N frames (configurable via crop_save_interval).

        Args:
            detections: The detections for the current frame
            frame: The raw image of the current frame
        """
        detections_track_ids = detections.tracker_id
        if detections_track_ids is None:
            return

        # Only save crops at specified intervals
        should_save_crop = (self.frame_count % self.crop_save_interval) == 0

        for i, tid in enumerate(detections_track_ids):
            class_id = detections.class_id[i] if detections.class_id is not None else -1
            confidence = detections.confidence[i] if detections.confidence is not None else 0.0

            if tid not in self.track_data:
                # For new tracks, always save the first crop
                crop = sv.crop_image(frame, detections.xyxy[i])
                self.track_data[tid] = TrackData(
                    track_id=tid,
                    crop=crop,
                    class_id=class_id,
                    confidence=confidence,
                    frame_id=self.frame_count,
                )
            else:
                # For existing tracks, only save crop at intervals
                if should_save_crop:
                    crop = sv.crop_image(frame, detections.xyxy[i])
                    self.track_data[tid].update_data(crop, class_id, confidence)
                else:
                    # Update metadata without crop
                    self.track_data[tid].update_metadata(class_id, confidence)

    def update_track_with_embedding_and_team(
        self, track_id: int, embedding: np.ndarray, team: int
    ) -> None:
        """
        Add an embedding and team information to the track data for a specific track ID.

        Args:
            track_id: The ID of the track to which the embedding belongs
            embedding: The embedding to be added
            team: The team ID associated with the track
        """
        if track_id in self.track_data:
            self.track_data[track_id].embedding = embedding
            self.track_data[track_id].team = team
        else:
            logger.warning(f"Track ID {track_id} not found in track data. Cannot add embedding.")

    def get_tracks_data(self) -> Dict[int, TrackData]:
        """
        Retrieve the track data for all tracks.

        Returns:
            A dictionary mapping track IDs to their corresponding TrackData objects.
        """
        return self.track_data

    def get_n_of_tracks(self) -> int:
        """
        Returns the number of unique tracks currently being tracked.

        Returns:
            The number of unique tracks.
        """
        return len(self.track_data)

    def create_embeddings_for_tracks(
        self, embeddings_processor: Callable, device: torch.device = torch.device("cpu")
    ) -> None:
        """
        Create embeddings for all tracks using batch processing.
        For each track, computes embeddings for all crops and averages them to create a single representative embedding.

        Args:
            embeddings_processor: A function that takes a batch of crops and returns embeddings
            device: The device to which the computation should be allocated. Defaults to CPU.
        """
        track_ids = list(self.track_data.keys())

        if not track_ids:
            logger.error("No track IDs found. Cannot create embeddings.")
            return

        logger.info(f"Creating embeddings for {len(track_ids)} tracks using batch processing.")

        total_tracks = len(track_ids)
        for idx, tid in enumerate(track_ids, 1):

            # ...existing code...
            crops = self.track_data[tid].crops
            if not crops:
                logger.warning(f"Track {tid} has no crops. Skipping embedding creation.")
                continue
            try:
                # Validate crops before processing (let transforms handle resizing)
                valid_crops = []
                for crop in crops:
                    # Basic validation - transforms will handle resizing
                    if crop is None or crop.size == 0 or len(crop.shape) < 2:
                        logger.warning(
                            f"Track {tid}: Skipping invalid crop (None, empty, or wrong dimensions)."
                        )
                        continue
                    # Check if crop has valid dimensions
                    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
                        logger.warning(
                            f"Track {tid}: Skipping crop with invalid dimensions {crop.shape}."
                        )
                        continue
                    # Check if crop is too small (minimum 1x1)
                    if crop.shape[0] < 1 or crop.shape[1] < 1:
                        logger.warning(f"Track {tid}: Skipping crop too small {crop.shape}.")
                        continue
                    valid_crops.append(crop)

                logger.debug(
                    f"Track {tid}: {len(valid_crops)}/{len(crops)} valid crops after filtering."
                )
                # Check if we have any valid crops after filtering
                if not valid_crops:
                    logger.warning(
                        f"Track {tid}: No valid crops found after filtering. Creating zero embedding."
                    )
                    self.track_data[tid].embedding = np.zeros(
                        (model_config.embedding_dim,), dtype=np.float32
                    )
                    continue

                # Pass crops directly to embeddings_processor (no manual preprocessing)
                # The embeddings_processor will handle transforms and normalization
                with torch.no_grad():
                    # Prefer to batch-process crops as a torch tensor on the target device
                    try:
                        if (
                            isinstance(valid_crops, list)
                            and len(valid_crops) > 0
                            and isinstance(valid_crops[0], np.ndarray)
                        ):
                            # Convert numpy crops to a single torch tensor batch
                            batch = [
                                torch.from_numpy(c).permute(2, 0, 1).float() for c in valid_crops
                            ]
                            batch = torch.stack(batch, dim=0).to(device)
                            track_embeddings = embeddings_processor(batch)
                        else:
                            track_embeddings = embeddings_processor(valid_crops)
                    except Exception:
                        # Fallback to original processor call (some processors expect PIL/ndarray inputs)
                        track_embeddings = embeddings_processor(valid_crops)

                # Convert to tensor if needed for averaging and normalization
                if isinstance(track_embeddings, np.ndarray):
                    track_embeddings = torch.tensor(
                        track_embeddings, device=device, dtype=torch.float32
                    )

                # Average the embeddings across all crops for this track
                avg_embedding = torch.mean(track_embeddings, dim=0)

                # Store the averaged embedding
                if isinstance(avg_embedding, torch.Tensor):
                    self.track_data[tid].embedding = avg_embedding.cpu().numpy()
                else:
                    self.track_data[tid].embedding = avg_embedding
                logger.debug(f"Track {tid}: Created embedding from {len(valid_crops)} valid crops.")
            except Exception as e:
                logger.error(f"Failed to create embedding for track {tid}: {e}")
                # Set a zero embedding as fallback (matching configured embedding dimension)
                self.track_data[tid].embedding = np.zeros(
                    (model_config.embedding_dim,), dtype=np.float32
                )

        logger.info(f"Successfully processed embeddings for {len(track_ids)} tracks.")

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the embeddings created for tracks.

        Returns:
            Dict containing statistics about tracks and embeddings
        """
        stats = {
            "total_tracks": len(self.track_data),
            "tracks_with_embeddings": 0,
            "tracks_without_embeddings": 0,
            "total_crops": 0,
            "crops_per_track": [],
            "embedding_dimensions": None,
        }

        for track_data in self.track_data.values():
            stats["total_crops"] += track_data.num_crops
            stats["crops_per_track"].append(track_data.num_crops)

            if track_data.embedding.size > 0:
                stats["tracks_with_embeddings"] += 1
                if stats["embedding_dimensions"] is None:
                    stats["embedding_dimensions"] = track_data.embedding.shape[0]
            else:
                stats["tracks_without_embeddings"] += 1

        if stats["crops_per_track"]:
            stats["avg_crops_per_track"] = np.mean(stats["crops_per_track"])
            stats["min_crops_per_track"] = np.min(stats["crops_per_track"])
            stats["max_crops_per_track"] = np.max(stats["crops_per_track"])

        return stats

    def update_tracks_with_loaded_detections(self, detections: sv.Detections, frame: np.ndarray):
        """
        Update the tracker objects with detections loaded from a JSON file.

        Args:
            detections: The loaded detections to update the tracker with.
            frame: The video frame corresponding to the detections.
        """
        self._update_tracks_obj(detections, frame)
