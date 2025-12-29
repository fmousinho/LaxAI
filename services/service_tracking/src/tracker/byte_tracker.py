import logging
logger = logging.getLogger(__name__)

from typing import Optional, List, Tuple

from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from shared_libs.config.transforms import TRANSFORMS
from schemas.tracking import TrackingParams
from tracker.kalman_filter import KalmanFilter, detection_np_tlbr_to_xyah
from tracker import matching
from tracker.basetrack import BaseTrack, TrackState

COMPENSATE_CAM_MOTION = True  # Affine transform to track camera motion signifantly improves accuracy

LOW_CONF_MAX_MATCH_DISTANCE = 0.9 # 1-IOU
UNCONFIRMED_MAX_MATCH_DISTANCE = 0.8

REID_UPDATE_ALPHA = 0.85

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.is_activated = False
        self.features: List[torch.Tensor] = []
        self.features_variance: Optional[torch.Tensor] = None
        self.features_count = 0

        self.score = score
        self.tracklet_len = 0
        self.predicted_mean: Optional[np.ndarray] = None
        self.match_cost: Optional[float] = 0.0
        self._initial_tlwh = np.asarray(tlwh, dtype=float)

    def predict(self):
        if self.mean is None or self.kalman_filter is None:
            return
        self.match_cost = None
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.predicted_mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.mean = self.predicted_mean.copy()

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].predicted_mean = mean
                stracks[i].covariance = cov
                stracks[i].match_cost = None

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.predicted_mean = self.mean
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, det_strack: 'STrack', frame_id: int, new_id: bool = False):
        if self.kalman_filter is None:
            raise ValueError("Kalman filter must be initialized before re-activation.")
        detection_xyah = det_strack.to_xyah()
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, detection_xyah)
        # self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = det_strack.score


    @staticmethod
    def multi_compensate_cam_motion(stracks, S_array, T_array):
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        multi_mean, multi_covariance = STrack.shared_kalman.multi_compensate_cam_motion(multi_mean, multi_covariance, S_array, T_array)
        for st, mean, covariance in zip(stracks, multi_mean, multi_covariance):
            st.mean = mean
            st.covariance = covariance

    def update(self, det_strack: 'STrack', frame_id: int):
        """
        Update a matched track
        :type det_strack: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        if self.kalman_filter is None:
            raise ValueError("Kalman filter must be initialized before update.")
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_xyah = det_strack.to_xyah()
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_xyah)
        self.state = TrackState.Tracked
        self.score = det_strack.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def predicted_tlbr(self):
        """Convert predicted state to bounding box format `(min x, min y, max x, max y)`."""
        if self.predicted_mean is None:
             return self.tlbr
        ret = self.predicted_mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def initial_tlwh(self):
        return self._initial_tlwh.copy()


    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args: TrackingParams, frame_rate=30, reid_model: Optional[torch.nn.Module] = None):
        self.tracked_stracks = []  # type: list[STrack]
        self.pending_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.ephemeral_stracks = []  # type: list[STrack]

        self.frame_id = 0
       
        self.track_activation_threshold = args.track_activation_threshold
        self.max_match_distance = args.max_match_distance
        self.prediction_threshold = args.prediction_threshold
        self.min_consecutive_frames = args.min_consecutive_frames
        self.lost_track_buffer = args.lost_track_buffer
        self.kalman_filter = KalmanFilter()

        self.embedding_update_frequency = args.embedding_update_frequency
        self.embedding_quality_threshold = args.embedding_quality_threshold
        self.embedding_min_detection_confidence = args.embedding_min_detection_confidence
        self.reid_model = reid_model
        self.reid_model_input_size = (224, 224)
        
        # Track prediction accuracy metrics
        self.prediction_errors = []  # Store all prediction errors for variance calculation
        self.observed_velocities = []  # Store observed velocities for empirical std_weight_velocity
   

    def update(self, detections_array: np.ndarray, S_array: Optional[np.ndarray], T_array: Optional[np.ndarray]) -> List[STrack]:
        """
        Args:
            detections_array: np.ndarray - The detections, expected to be in the format N x [x1, y1, x2, y2, score]
            S_array: Optional[np.ndarray] - The scale array for the Kalman filter.
            T_array: Optional[np.ndarray] - The T array for the Kalman filter.
            The arrays update the current state of the filter by K' = K * S_array + T_array
        
        Returns:
            A list of stracks
        """

        logger.debug("")
        logger.debug(f"FRAME {self.frame_id} UPDATE")
        
       
        S_array = S_array.copy() if S_array is not None else None
        T_array = T_array.copy() if T_array is not None else None
        
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        pending_stracks = []
        removed_stracks = []
        ephemeral_stracks = []
        # Apply detection filtering (border removal + ambiguous rejection)
        from tracking_utils.detection_filters import remove_border_detections
        detections = remove_border_detections(detections_array, frame_size=self.frame_size, margin=2)

        if detections.shape[1] >= 5:
            scores = detections[:, 4]
        else:
            raise ValueError("Detections array must have at least 5 columns: [x1, y1, x2, y2, score]")

        remain_inds_mask = np.where(scores >= self.prediction_threshold)[0]
        remain_detections = detections[remain_inds_mask]

        low_conf_mask = np.where(remain_detections[:,4] < self.track_activation_threshold)[0]
        high_conf_mask = np.where(remain_detections[:,4] >= self.track_activation_threshold)[0]

        detections_high = remain_detections[high_conf_mask]
        detections_low = remain_detections[low_conf_mask]

        logger.debug(f"total detections: {len(remain_detections)} | thresh: {self.track_activation_threshold} | high conf: {len(high_conf_mask)} | low: {len(low_conf_mask)}")

        strack_pool = joint_stracks(self.tracked_stracks, self.lost_stracks)
        
        # Reject ambiguous detections that overlap with multiple tracks
        if len(strack_pool) > 0 and len(remain_detections) > 0:
            from tracking_utils.detection_filters import reject_ambiguous_detections
            track_predictions = np.array([t.predicted_tlbr for t in strack_pool])
            remain_detections = reject_ambiguous_detections(remain_detections, track_predictions, iou_threshold=0.4)
            
            # Re-split into high and low confidence after rejection
            low_conf_mask = np.where(remain_detections[:,4] < self.track_activation_threshold)[0]
            high_conf_mask = np.where(remain_detections[:,4] >= self.track_activation_threshold)[0]
            detections_high = remain_detections[high_conf_mask]
            detections_low = remain_detections[low_conf_mask]
            logger.debug(f"After ambiguous rejection: {len(remain_detections)} detections | high: {len(high_conf_mask)} | low: {len(low_conf_mask)}")

        # Camera motion compensation, if arrays are provided and valid
        if len(strack_pool) > 0 and S_array is not None and T_array is not None and COMPENSATE_CAM_MOTION:
            # Check for NaNs or Infs which cause RuntimeWarnings
            if np.isfinite(S_array).all() and np.isfinite(T_array).all():
                STrack.multi_compensate_cam_motion(strack_pool, S_array, T_array)
            else:
                logger.warning(f"Frame {self.frame_id}: Invalid camera motion arrays (NaN/Inf detected). Skipping compensation.")

        STrack.multi_predict(strack_pool)

        ''' Step 1: First association, with high score detection boxes'''
        logger.debug(f"=== Step 1: Association with high score detections ===")
        (newly_activated_stracks, newly_refind_stracks, newly_pending_stracks, unmatched_stracks, unmatched_detections_s1) = self.association_step(
            strack_pool, detections_high, max_match_distance=0.8
        )
        activated_stracks.extend(newly_activated_stracks)
        refind_stracks.extend(newly_refind_stracks)
        pending_stracks.extend(newly_pending_stracks)

        ''' Step 2: Second association, with low score detection boxes'''
        logger.debug(f"=== Step 2: Association with low score detections ====")
        (newly_activated_stracks, newly_refind_stracks, newly_pending_stracks, unmatched_stracks, unmatched_detections_s2) = self.association_step(
            unmatched_stracks, detections_low, max_match_distance=0.5
        )
        activated_stracks.extend(newly_activated_stracks)
        refind_stracks.extend(newly_refind_stracks)
        pending_stracks.extend(newly_pending_stracks)

        # Step 3 uses unmatched detections from both Step 1 and Step 2
        unmatched_detections = np.concatenate((unmatched_detections_s1, unmatched_detections_s2), axis=0) 

        for track in unmatched_stracks:
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """ Step 3: Deal with unconfirmed tracks """
        logger.debug(f"=== Step 3: Deal with unconfirmed tracks =============")
        (newly_activated_stracks, _, newly_pending_stracks, unmatched_stracks, unmatched_detections) = self.association_step(
            self.pending_stracks, unmatched_detections, max_match_distance=0.8
        )
        activated_stracks.extend(newly_activated_stracks)
        pending_stracks.extend(newly_pending_stracks)

        for track in unmatched_stracks:
            track.mark_ephemeral()
            ephemeral_stracks.append(track)

        """ Step 4: Create new tracks """
        logger.debug(f"=== Step 4: Create new tracks ========================")
        for detection in unmatched_detections:
            bbox = bboxes_from(detection[np.newaxis, :])[0]
            score = scores_from(detection[np.newaxis, :])[0]
            if score < self.prediction_threshold:
                continue
            track = STrack(STrack.tlbr_to_tlwh(bbox), score)
            track.activate(self.kalman_filter, self.frame_id)
            if track.score >= self.embedding_quality_threshold:
                self.initiate_track_reid_features(track)
            pending_stracks.append(track)
            self.update_with_track(detection, track)
            bbox_str = ", ".join(f"{int(x):4d}" for x in bbox)
            logger.debug(f"New track {track.track_id:4d} created with score {score:.4f}, TLBR [{bbox_str}]")

        """ Step 5: Update state for lost tracks"""
        logger.debug(f"=== Step 5: Update state for lost tracks =============")
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.lost_track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
                pred_tlbr_str = ", ".join(f"{int(x):4d}" for x in track.predicted_tlbr)
                logger.debug(f"Track {track.track_id:4d} removed, pred TLBR [{pred_tlbr_str}]")
            elif track.state == TrackState.Lost:
                lost_stracks.append(track)
                pred_tlbr_str = ", ".join(f"{int(x):4d}" for x in track.predicted_tlbr)
                logger.debug(f"Track {track.track_id:4d} lost, pred TLBR [{pred_tlbr_str}]")

        self.tracked_stracks = activated_stracks + refind_stracks
        self.lost_stracks = lost_stracks
        self.pending_stracks = pending_stracks
        self.removed_stracks.extend(removed_stracks)
        self.ephemeral_stracks.extend(ephemeral_stracks)
        
        # Update embeddings for confirmed tracks
        all_tracks_for_overlap = self.tracked_stracks + self.lost_stracks
        for track in self.tracked_stracks:
            if track.is_activated:
                self.update_track_reid_features(track, all_tracks_for_overlap, detections)
        
        TRACK_BY_TRACK_LOG = False 
        if logger.getEffectiveLevel() == logging.DEBUG and TRACK_BY_TRACK_LOG:
            self.log_frame_summary()

        # Return only confirmed tracks
        output_stracks = self.tracked_stracks

        self.frame_id += 1
        return output_stracks

    def association_step(self, 
                         strack_pool: List[STrack], 
                         detections: np.ndarray, 
                         max_match_distance: float = 0.8,
    ) -> Tuple[List[STrack], List[STrack], List[STrack], List[STrack], List[np.ndarray]]:
        """
        Performs an association step: cost calculation, matching, and track updates.
        Returns: (activated_stracks, refind_stracks, unmatched_tracks, unmatched_detections)
        """
        activated_stracks = []
        pending_stracks = []
        refind_stracks = []
        unmatched_tracks = []
        unmatched_detections = []
        
        if len(strack_pool) > 0 and len(detections) > 0:
            bboxes = bboxes_from(detections)
            scores = scores_from(detections)
            
            # Compute full association cost with all penalties and gates
            cost_matrix = matching.compute_association_cost(
                strack_pool,
                detections,
                kalman_filter_obj=None,  # Can be added later if needed
                apply_aspect_ratio_penalty=True,
                apply_height_gate=True,
                aspect_ratio_factor=0.6,
                height_threshold=0.2,
            )
            matches, u_track, u_detection = matching.linear_assignment(cost_matrix, thresh=max_match_distance)
            
            for itracked, idet in matches:
                strack = strack_pool[itracked]
                bbox = bboxes[idet]
                score = scores[idet]    
                strack.match_cost = cost_matrix[itracked, idet]
                
                # Calculate prediction error (difference between predicted and actual TLBR)
                pred_tlbr = strack.predicted_tlbr
                actual_tlbr = bbox
                error = np.abs(pred_tlbr - actual_tlbr)  # Element-wise absolute difference
                self.prediction_errors.append(error)
                
                # Calculate observed velocity (change in position from last update)
                # The Kalman state has velocities at indices 4-7: [vx, vy, va, vh]
                # We can extract the actual velocity from the state after update
                if strack.mean is not None:
                    # Store velocity components (vx, vy, vh) - skip aspect ratio velocity
                    observed_vel = np.array([
                        strack.mean[4],  # vx (center x velocity)
                        strack.mean[5],  # vy (center y velocity)
                        strack.mean[7]   # vh (height velocity)
                    ])
                    self.observed_velocities.append(observed_vel)
                
                self.update_with_track(detections[idet], strack)
                
                if strack.state == TrackState.Tracked:
                    strack.update(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id)
                    if strack.tracklet_len >= self.min_consecutive_frames:
                        strack.is_activated = True
                        activated_stracks.append(strack)
                    else:
                        pending_stracks.append(strack)
                else:
                    strack.re_activate(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id, new_id=False)
                    refind_stracks.append(strack)
                bbox_str = ", ".join(f"{int(x):4d}" for x in bbox)
                pred_tlbr_str = ", ".join(f"{int(x):4d}" for x in strack.predicted_tlbr)
                logger.debug(f"Track {strack.track_id:4d} matched with cost {cost_matrix[itracked, idet]:.4f}, conf {score:.4f}, TLBR [{bbox_str}], pred TLBR [{pred_tlbr_str}]")

            unmatched_tracks = [strack_pool[i] for i in u_track]
            if logger.getEffectiveLevel() == logging.DEBUG:
                for i in u_track:
                    pred_tlbr_str = ", ".join(f"{int(x):4d}" for x in strack_pool[i].predicted_tlbr)
                    logger.debug(f"Track {strack_pool[i].track_id:4d} NOT matched, best cost {cost_matrix[i].min():.4f}, pred TLBR [{pred_tlbr_str}]") 
            
            unmatched_detections = detections[u_detection]
            if logger.getEffectiveLevel() == logging.DEBUG:
                for i in u_detection:
                    tlbr_str = ", ".join(f"{int(x):4d}" for x in detections[i, :4])
                    logger.debug(f"Detection NOT matched, best cost {cost_matrix[:, i].min():.4f}, conf {detections[i, 4]:.4f}, TLBR [{tlbr_str}]")

            return activated_stracks, refind_stracks, pending_stracks, unmatched_tracks, unmatched_detections
        else:
            return [], [], [], strack_pool, detections

    def log_frame_summary(self):
        """
        Log a summary of all tracks in the current frame, sorted by track ID.
        """
        # Deduplicate tracks across lists to avoid duplicate IDs in the log
        all_tracks = joint_stracks(
            joint_stracks(self.tracked_stracks, self.lost_stracks),
            self.removed_stracks
        )
        # Sort by track_id
        all_tracks.sort(key=lambda x: x.track_id)

        statuses_for_metrics = ["Tracked", "Pending", "Lost"]
        logger.debug(f"=== Frame {self.frame_id} Summary ===")
        for t in all_tracks:
            # Determine status string
            status = "Unknown"
            if t.state == TrackState.Tracked:
                if t.is_activated:
                    status = "Tracked"
                else:
                    status = "Pending"
            elif t.state == TrackState.Lost:
                status = "Lost"
            elif t.state == TrackState.Removed and not t.is_activated:
                status = "Ephemeral"
            elif t.state == TrackState.Removed and t.is_activated:
                status = "Removed"
            
            if status in statuses_for_metrics or status == "Removed":
                pred_tlbr = t.predicted_tlbr
                curr_tlbr = t.tlbr
                pred_str = f"[{pred_tlbr[0]:4.0f}, {pred_tlbr[1]:3.0f}, {pred_tlbr[2]:4.0f}, {pred_tlbr[3]:3.0f}]" 
                curr_str = f"[{curr_tlbr[0]:4.0f}, {curr_tlbr[1]:3.0f}, {curr_tlbr[2]:4.0f}, {curr_tlbr[3]:3.0f}]"
                cost_str = f"{t.match_cost:.4f}" if t.match_cost is not None else "N/A"
                if status == "Removed":
                    cost_str = "REMOVED"
                score_str = f"{t.score:.2f}"
            else:
                pred_str = None
                curr_str = None
                cost_str = None
                score_str = None
                pred_tlbr = None
                curr_tlbr = None

            logger.debug(
                f"ID: {t.track_id:3d} | Status: {status:9s} | "
                f"Score: {score_str} | Cost: {cost_str} | "
                f"Pred TLBR: {pred_str} | TLBR: {curr_str}"
            )


    def assign_tracks_to_detections(
        self, detections_array: np.ndarray, S_array: Optional[np.ndarray], T_array: Optional[np.ndarray], frame: np.ndarray
        ) -> np.ndarray:
        """
        Update the tracker with new detections and camera motion compensation, and returns the detection array with updated tracks.
        
        This method is optimized to avoid redundant matching by tracking detection-to-track assignments during the update process.
        
        Args:
            detections_array: Array of detections in the format (N, 5) where N is the number of detections.
            S_array: Array of scale transformations for camera motion compensation.
            T_array: Array of translation transformations for camera motion compensation.
            frame: Frame to extract features from.
        
        Returns:
            np.ndarray: Array with dimensions (N, 6) with updated tracks, with the format [x1, y1, x2, y2, score, track_id].
            Unknown detections are returned with track_id = -1.
        """
        # Prepare 7-column array: [x1, y1, x2, y2, score, track_id, original_index, is_activated]
        n = len(detections_array)
        detections = np.zeros((n, 8))
        detections[:, :5] = detections_array[:, :5]
        detections[:, 5] = -1
        detections[:, 6] = np.arange(n)

        self.frame_size = (frame.shape[1], frame.shape[0])
        self.frame_size = (frame.shape[1], frame.shape[0])
        self._frame = frame
        # Store reference for update_with_track
        self._original_detections = detections
        
        _ = self.update(detections, S_array, T_array)

        confirmed_tracks_only = True
        if confirmed_tracks_only:
            self._original_detections[:, 5][self._original_detections[:, 7] == 0] = -1 
 
        return self._original_detections[:, :6]

    def log_prediction_statistics(self):
        """
        Compute and log variance statistics for prediction errors and observed velocities.
        This should be called at the end of video processing.
        """
        if len(self.prediction_errors) == 0:
            logger.info("No prediction errors recorded")
            return
        
        # Convert list to numpy array for easier computation
        errors_array = np.array(self.prediction_errors)  # Shape: (N, 4) where 4 is [x1, y1, x2, y2]
        
        # Compute statistics for each coordinate
        mean_errors = np.mean(errors_array, axis=0)
        variance_errors = np.var(errors_array, axis=0)
        std_errors = np.std(errors_array, axis=0)
        
        # Overall statistics (average across all coordinates)
        overall_mean = np.mean(mean_errors)
        overall_variance = np.mean(variance_errors)
        overall_std = np.mean(std_errors)
        
        logger.info("=" * 80)
        logger.info("PREDICTION ACCURACY STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total matches analyzed: {len(self.prediction_errors)}")
        logger.info("")
        logger.info("Per-coordinate statistics (x1, y1, x2, y2):")
        logger.info(f"  Mean error:     [{mean_errors[0]:6.2f}, {mean_errors[1]:6.2f}, {mean_errors[2]:6.2f}, {mean_errors[3]:6.2f}]")
        logger.info(f"  Std deviation:  [{std_errors[0]:6.2f}, {std_errors[1]:6.2f}, {std_errors[2]:6.2f}, {std_errors[3]:6.2f}]")
        logger.info(f"  Variance:       [{variance_errors[0]:6.2f}, {variance_errors[1]:6.2f}, {variance_errors[2]:6.2f}, {variance_errors[3]:6.2f}]")
        logger.info("")
        logger.info(f"Overall statistics:")
        logger.info(f"  Mean error:     {overall_mean:6.2f} pixels")
        logger.info(f"  Std deviation:  {overall_std:6.2f} pixels")
        logger.info(f"  Variance:       {overall_variance:6.2f} pixels²")
        logger.info("=" * 80)
        
        # Compute velocity statistics
        if len(self.observed_velocities) > 0:
            velocities_array = np.array(self.observed_velocities)  # Shape: (N, 3) where 3 is [vx, vy, vh]
            
            # Compute statistics for each velocity component
            mean_velocities = np.mean(velocities_array, axis=0)
            std_velocities = np.std(velocities_array, axis=0)
            variance_velocities = np.var(velocities_array, axis=0)
            
            # Get typical height from the data (for normalization)
            # We'll use the mean height from all tracks
            typical_height = np.mean([t.tlbr[3] - t.tlbr[1] for t in self.tracked_stracks + self.lost_stracks if hasattr(t, 'tlbr')])
            
            # Calculate empirical std_weight_velocity
            # The Kalman filter uses: std_vel = std_weight_velocity * height
            # So: std_weight_velocity = std_vel / height
            # We want the standard deviation of velocity normalized by height
            empirical_std_weight_vx = std_velocities[0] / typical_height if typical_height > 0 else 0
            empirical_std_weight_vy = std_velocities[1] / typical_height if typical_height > 0 else 0
            empirical_std_weight_vh = std_velocities[2] / typical_height if typical_height > 0 else 0
            
            # Average across x and y (position velocities)
            empirical_std_weight_velocity = (empirical_std_weight_vx + empirical_std_weight_vy) / 2.0
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("VELOCITY STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Total velocity samples: {len(self.observed_velocities)}")
            logger.info(f"Typical object height:  {typical_height:.2f} pixels")
            logger.info("")
            logger.info("Per-component statistics (vx, vy, vh):")
            logger.info(f"  Mean velocity:  [{mean_velocities[0]:6.2f}, {mean_velocities[1]:6.2f}, {mean_velocities[2]:6.2f}] px/frame")
            logger.info(f"  Std deviation:  [{std_velocities[0]:6.2f}, {std_velocities[1]:6.2f}, {std_velocities[2]:6.2f}] px/frame")
            logger.info(f"  Variance:       [{variance_velocities[0]:6.2f}, {variance_velocities[1]:6.2f}, {variance_velocities[2]:6.2f}] (px/frame)²")
            logger.info("")
            logger.info("Empirical std_weight_velocity (normalized by height):")
            logger.info(f"  vx component:   {empirical_std_weight_vx:.6f}")
            logger.info(f"  vy component:   {empirical_std_weight_vy:.6f}")
            logger.info(f"  vh component:   {empirical_std_weight_vh:.6f}")
            logger.info(f"  RECOMMENDED:    {empirical_std_weight_velocity:.6f}  (average of vx, vy)")
            logger.info("")
            logger.info(f"Current std_weight_velocity in KalmanFilter: {self.kalman_filter._std_weight_velocity:.6f}")
            logger.info("=" * 80)


    def initiate_track_reid_features(self, track: STrack):
        """
        Uses class reid model to extract features from track bounding box
        """
        feats = self.get_embeddings(track.tlbr)
        if feats is not None:
            track.features = [feats]
            track.features_count = 1

    def update_track_reid_features(self, track: STrack, all_tracks: List[STrack], all_detections: np.ndarray):
        """
        Updates the track's reid features using the given detection bounding box.
        Updates every N frames.
        """
        if track.score < self.embedding_quality_threshold:
            return
        
        if track.features is None:
            self.initiate_track_reid_features(track)
            return
        
        if track.tracklet_len % self.embedding_update_frequency != 0:
            return

        # Overlap Check
        current_bbox = track.tlbr
        
        # 1. Check vs other detections
        if len(all_detections) > 0:
             ious_dets = matching.bbox_ious(
                 np.ascontiguousarray(all_detections[:, :4], dtype=float), 
                 current_bbox[np.newaxis, :]
            )
             # Filter out the detection that likely belongs to this track (highest IoU/closest)
             # But simplistic check: if ANY other detection has IoU > 0, it's risky.
             # Actually, we need to exclude the detection matched to THIS track.
             # Since we don't have the matched detection index easily passed here without refactoring,
             # we can assume "significant" overlap is bad.
             # If IoU > 0.05 with MORE than one detection (itself), it's ambiguous.
             # Or, since we are in `update`, the track IS associated.
             # Let's count how many detections have IoU > 0.
             count_overlaps = np.sum(ious_dets > 0.0)
             if count_overlaps > 1: # Itself + another
                 return 

        # 2. Check vs other tracks (projected)
        # We need to exclude 'self' from the list
        other_tracks = [t for t in all_tracks if t.track_id != track.track_id]
        if len(other_tracks) > 0:
            other_tlbrs = np.array([t.tlbr for t in other_tracks], dtype=float)
            ious_tracks = matching.bbox_ious(other_tlbrs, current_bbox[np.newaxis, :])
            if np.any(ious_tracks > 0.0):
                return
       
        new_feat = self.get_embeddings(track.tlbr)
        
        if new_feat is not None:
            # Append new feature to list
            track.features.append(new_feat)
            track.features_count += 1
            

    def get_embeddings(self, tlbr: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extracts embeddings for a given bounding box (tlbr).
        """
        # Need access to frame. We stored it in _original_detections context or need to store it in self._frame
        # In the original file, self._frame wasn't stored in `assign_tracks_to_detections`. 
        # I need to ensure self._frame is available. 
        # `assign_tracks_to_detections` sets `self.frame_size` but not `self._frame`.
        # I need to modify `assign_tracks_to_detections` to store `self._frame`.
        
        if not hasattr(self, '_frame') or self._frame is None:
             # Fallback if frame is not available (should not happen if flow is correct)
             return None
             
        if self.reid_model is None:
             return None # Can't extract
        
        img_h, img_w = self._frame.shape[:2]
        
        x1 = max(0, int(tlbr[0]))
        y1 = max(0, int(tlbr[1]))
        x2 = min(img_w, int(tlbr[2]))
        y2 = min(img_h, int(tlbr[3]))
        
        if x2 <= x1 or y2 <= y1:
            return None

        crop = self._frame[y1:y2, x1:x2]
        transform = TRANSFORMS["inference_for_non_pil"]
        
        crop_tensor = transform(crop)
        crop_tensor = crop_tensor.unsqueeze(0) # Add batch dimension (C, H, W) -> (1, C, H, W)

        # Ensure tensor is on the same device as the model
        device = next(self.reid_model.parameters()).device
        crop_tensor = crop_tensor.to(device)

        with torch.no_grad():
            return self.reid_model(crop_tensor).detach()

    def update_with_track(self, detection: np.ndarray, track: STrack):
        """
        Updates the detection to track mapping based on the current tracks.
        
        Args:
            detection (np.ndarray): The detection to update (1, :)
            track (STrack): The track to update.
        """
        if len(detection.shape) > 1:
            raise ValueError("Detection must be a 1D array")
        
        # Update local copy (though often a copy, useful if reused in same scope)
        detection[5] = track.track_id
        
        # Update original array if available
        if hasattr(self, '_original_detections') and self._original_detections is not None:
             # Ensure we don't go out of bounds if detection doesn't have 7th column (though it should)
             if detection.shape[0] >= 8:
                 idx = int(detection[6])
                 self._original_detections[idx, 5] = track.track_id
                 self._original_detections[idx, 7] = 1 if track.is_activated else 0

        

def joint_stracks(tlista, tlistb):
    """
    Join two track lists, ensuring unique track IDs.
    Retains the instance from the FIRST list if duplicates exist.
    """
    tmp = {t.track_id: t for t in tlistb}
    tmp.update({t.track_id: t for t in tlista})
    return list(tmp.values())


def sub_stracks(tlista, tlistb):
    """
    Remove tracks in tlistb from tlista based on track ID.
    """
    exclude_ids = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in exclude_ids]


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.add(q)
        else:
            dupa.add(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb

def remove_border_detections_from(detections: np.ndarray, frame_size: Tuple[int, int] = (1920, 1080), margin: int = 2) -> np.ndarray:
    """
    Removes detections that touch the border to improve embedding quality. Modifies the original array in place.

    Args:
        detections (np.ndarray): Array of detections with shape (N, 5) where N is the number of detections.
        frame_size (Tuple[int, int]): Tuple of (width, height) of the frame.
        margin (int): Number of pixels to leave from the border.
    """
    if len(detections) > 0:
        img_w, img_h = frame_size
        bboxes_remain = detections[:, :4]
        not_touching_border = (
                (bboxes_remain[:, 0] > margin) & 
                (bboxes_remain[:, 1] > margin) & 
                (bboxes_remain[:, 2] < img_w - margin) & 
                (bboxes_remain[:, 3] < img_h - margin)
            )
        return detections[not_touching_border]

def panalize_aspect_ratio_swings(cost_matrix: np.ndarray, stracks: List[STrack], detections: np.ndarray, factor: float = .5):
    if len(stracks) != cost_matrix.shape[0] or len(detections) != cost_matrix.shape[1]:
        raise ValueError("Cost matrix dimensions do not match the number of tracks and detections")
    cost_matrix += matching.aspect_ratio_distance(stracks, detections) * factor

def penalize_scale_swings(cost_matrix: np.ndarray, stracks: List[STrack], detections: np.ndarray, factor: float = 1.0):
    if len(stracks) != cost_matrix.shape[0] or len(detections) != cost_matrix.shape[1]:
        raise ValueError("Cost matrix dimensions do not match the number of tracks and detections")
    cost_matrix += matching.scale_distance(stracks, detections) * factor

def bboxes_from(detections: np.ndarray) -> np.ndarray:
    return detections[:, :4]
        
def scores_from(detections: np.ndarray) -> np.ndarray:
    return detections[:, 4]

def gate_cost_matrix_height(cost_matrix: np.ndarray, tracks: List, detections: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Gate association if height change is too large.
    :param cost_matrix:
    :param tracks: list[STrack]
    :param detections: np.ndarray (N, >=4) in tlbr format
    :param threshold: max relative change in height
    :return: cost_matrix
    """
    if cost_matrix.size == 0 or len(tracks) == 0 or len(detections) == 0:
        return cost_matrix

    detections_tlbr = detections[:, :4]
    det_heights = detections_tlbr[:, 3] - detections_tlbr[:, 1]
    
    # We use track.tlbr which computes it from mean
    track_heights = np.array([(t.tlbr[3] - t.tlbr[1]) for t in tracks])
    
    # Expand for broadcasting
    # tracks: (N, 1), dets: (1, M)
    t_h = track_heights[:, np.newaxis]
    d_h = det_heights[np.newaxis, :]
    
    # Check relative difference: abs(t - d) / t
    # Add epsilon to avoid div by zero
    diff = np.abs(t_h - d_h) / (t_h + 1e-6)
    
    cost_matrix[diff > threshold] = np.inf
    return cost_matrix

