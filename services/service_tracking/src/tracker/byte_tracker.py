import logging
logger = logging.getLogger(__name__)

from typing import Optional, List

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
from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState

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
        self.features: Optional[torch.Tensor] = None

        self.score = score
        self.tracklet_len = 0
        self.predicted_mean: Optional[np.ndarray] = None
        self.match_cost: Optional[float] = 0.0

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

    def re_activate(self, new_track, frame_id, new_id=False):
        if self.kalman_filter is None:
            raise ValueError("Kalman filter must be initialized before re-activation.")
        
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score


    @staticmethod
    def multi_compensate_cam_motion(stracks, S_array, T_array):
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        multi_mean, multi_covariance = STrack.shared_kalman.multi_compensate_cam_motion(multi_mean, multi_covariance, S_array, T_array)
        for st, mean, covariance in zip(stracks, multi_mean, multi_covariance):
            st.mean = mean
            st.covariance = covariance

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        if self.kalman_filter is None:
            raise ValueError("Kalman filter must be initialized before update.")
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.score = new_track.score


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
        self.reid_enabled = reid_model is not None
        if self.reid_enabled:
            self.reid_model_input_size = (224, 224)
            self.reid_model = reid_model

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
        
        self.frame_id += 1
        detections_array = detections_array.copy()
        S_array = S_array.copy() if S_array is not None else None
        T_array = T_array.copy() if T_array is not None else None
        
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        ephemeral_stracks = []

        if detections_array.shape[1] >= 5:
            scores = detections_array[:, 4]
            bboxes = detections_array[:, :4]
        else:
            raise ValueError("Detections array must have at least 5 columns: [x1, y1, x2, y2, score]")

        remain_inds_mask = np.where(scores >= self.prediction_threshold)[0]
        logger.debug(f"FRAME {self.frame_id}")

        # All detections indexes are based on the "keep" array
        scores_keep = scores[remain_inds_mask]
        bboxes_keep = bboxes[remain_inds_mask]
        
        low_conf_mask = np.where(scores_keep < self.track_activation_threshold)[0]
        high_conf_mask = np.where(scores_keep >= self.track_activation_threshold)[0]

        bboxes_low = bboxes_keep[low_conf_mask]
        bboxes_high = bboxes_keep[high_conf_mask]
        scores_low = scores_keep[low_conf_mask]
        scores_high = scores_keep[high_conf_mask]

        n_tracks_reconfirmed = 0  # Already tracked and "is_activated", updated with new detection
        n_tracks_confirmed = 0  # Already tracked and just became "is_active" because of number of consecutive detections
        n_tracks_new = 0  # New tracklet
        n_tracks_pending_confirmation = 0 #  Hasn't reached the number of consecutive detections to become is_activated = True yet

        logger.debug(f"total detections: {len(scores_keep)} | thresh: {self.track_activation_threshold} | high conf: {len(high_conf_mask)} | low: {len(low_conf_mask)}")

        # Add newly detected tracklets to tracked_stracks
        pending_confirmation_tracks = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                pending_confirmation_tracks.append(track)
            else:
                tracked_stracks.append(track)
       #  logger.debug(f"is_activated stracks before association: {len(tracked_stracks)} | pending conf: {len(pending_confirmation_tracks)} | lost: {len(self.lost_stracks)}")

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Camera motion compensation, if arrays are provided and valid
        if len(strack_pool) > 0 and S_array is not None and T_array is not None and COMPENSATE_CAM_MOTION:
            # Check for NaNs or Infs which cause RuntimeWarnings
            if np.isfinite(S_array).all() and np.isfinite(T_array).all():
                STrack.multi_compensate_cam_motion(strack_pool, S_array, T_array)
            else:
                logger.warning(f"Frame {self.frame_id}: Invalid camera motion arrays (NaN/Inf detected). Skipping compensation.")

        STrack.multi_predict(strack_pool)

        ''' Step 1: First association, with high score detection boxes'''
        if len(strack_pool) > 0 and len(bboxes_high) > 0:
            if self.reid_enabled:
                dists = matching.v_iou_reid_distance(strack_pool, bboxes_high, self.get_embeddings, iou_thresh=.3, reid_weight=.5)
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=.9)
            else:
                dists = matching.v_iou_distance(strack_pool, bboxes_high)
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.max_match_distance)
        else:
            matches = []
            u_track = np.arange(len(strack_pool))
            u_detection = np.arange(len(bboxes_high))
            dists = None

        unmatched_tracks = [strack_pool[i] for i in u_track]

        for itracked, idet in matches:
            track = strack_pool[itracked]
            bbox = bboxes_high[idet]
            score = scores_high[idet]
            track.match_cost = dists[itracked, idet]
            self._detection_to_track_map[high_conf_mask[idet]] = track.track_id
            
            if track.state == TrackState.Tracked:
                track.update(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id)
                activated_stracks.append(track)
                n_tracks_reconfirmed += 1
            else:
                track.re_activate(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id, new_id=False)
                refind_stracks.append(track)

            if self.reid_enabled:
                self.update_track_reid_features(track)

        self.step_log("step 1", matches, unmatched_tracks, u_detection, dists)

        ''' Step 2: Second association, with low score detection boxes'''
        if len(unmatched_tracks) > 0 and len(bboxes_low) > 0:
            if self.reid_enabled:
                dists = matching.v_iou_reid_distance(unmatched_tracks, bboxes_low, self.get_embeddings, iou_thresh=.4, reid_weight=.5)
                matches, u_track_second, u_detection_second = matching.linear_assignment(dists, thresh=.8) # thresh: cost
            else:
                dists = matching.v_iou_distance(unmatched_tracks, bboxes_low)
                matches, u_track_second, u_detection_second = matching.linear_assignment(dists, thresh=.5)
        else:
            matches = []
            u_track_second = np.arange(len(unmatched_tracks))
            u_detection_second = np.arange(len(bboxes_low))
            dists = None

        unmatched_tracks_second = [unmatched_tracks[i] for i in u_track_second]

        for itracked, idet in matches:
            track = unmatched_tracks[itracked]
            bbox = bboxes_low[idet]
            score = scores_low[idet]
            track.match_cost = dists[itracked, idet]
            self._detection_to_track_map[low_conf_mask[idet]] = track.track_id

            if track.state == TrackState.Tracked:
                track.update(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id)
                activated_stracks.append(track)
                n_tracks_reconfirmed += 1
            else:
                track.re_activate(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id, new_id=False)
                refind_stracks.append(track)

        for track in unmatched_tracks_second:
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        self.step_log("step 2", matches, unmatched_tracks, u_detection_second, dists)
 
        '''Step 3: Deal with pending confirmation tracks'''
        unmatched_detections_mask = np.concatenate((high_conf_mask[u_detection], low_conf_mask[u_detection_second]), axis=0)
        unmatched_bbox_array = bboxes_keep[unmatched_detections_mask]
        unmatched_scores_array = scores_keep[unmatched_detections_mask]

        if len(pending_confirmation_tracks) > 0 and len(unmatched_bbox_array) > 0:
            dists = matching.v_iou_distance(pending_confirmation_tracks, unmatched_bbox_array)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=UNCONFIRMED_MAX_MATCH_DISTANCE)
        else:
            matches = []
            u_unconfirmed = np.arange(len(pending_confirmation_tracks))
            u_detection = np.arange(len(unmatched_bbox_array))
            dists = None

        for itracked, idet in matches:
            track = pending_confirmation_tracks[itracked]
            bbox = unmatched_bbox_array[idet]
            score = unmatched_scores_array[idet]
            track.match_cost = dists[itracked, idet]
            track.update(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id)
            
            if track.tracklet_len >= self.min_consecutive_frames:
                track.is_activated = True
                activated_stracks.append(track)
                n_tracks_confirmed += 1
            else:
                activated_stracks.append(track) # Unconfirmed tracks are part of the activated_stracks list, but are not "is_activated"
                n_tracks_pending_confirmation += 1
                 
            self._detection_to_track_map[unmatched_detections_mask[idet]] = track.track_id

        self.step_log("step 3", matches, unmatched_tracks, u_detection, dists)

        unmatched_detections_mask = unmatched_detections_mask[u_detection]
        unmatched_detections_array = np.column_stack((bboxes_keep[unmatched_detections_mask], scores_keep[unmatched_detections_mask])) if len(unmatched_detections_mask) > 0 else np.empty((0, 5))
        unconfirmed_tracks_remaining = [pending_confirmation_tracks[i] for i in u_unconfirmed]

        # Tracks pending confirmation that are not matched right away are removed
        for track in unconfirmed_tracks_remaining:
            track.mark_ephemeral()
            ephemeral_stracks.append(track)

        """ Step 4: Deal with unmatched detections"""

        """ 4.1: Scavenger for removed tracks (reid mode only)"""
      
        if self.reid_enabled:
            
            #  Detections in either side of the frame are more likely to be from a previous track
            scavenged_indices = set()
            if len(self.removed_stracks) > 0 and len(unmatched_detections_array) > 0:
                dists = matching.v_iou_reid_distance(self.removed_stracks, unmatched_detections_array, self.get_embeddings, iou_thresh=0.0, reid_weight=1.0)
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=.8)
                for itracked, idet in matches:
                    scavenged_indices.add(itracked)
                    track = self.removed_stracks[itracked]
                    bbox = unmatched_detections_array[idet, :4]
                    score = unmatched_detections_array[idet, 4]
                    track.match_cost = dists[itracked, idet]
                    track.mark_scavenged()
                    track.update(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id)
                    activated_stracks.append(track)
                    self._detection_to_track_map[unmatched_detections_mask[idet]] = track.track_id

            else:
                u_detection = np.arange(len(unmatched_detections_array))
                u_track = np.arange(len(self.removed_stracks))
                matches = []
                dists = None

            self.step_log("step 4.1", matches, unmatched_tracks, u_detection, dists)

            unmatched_detections_mask = unmatched_detections_mask[u_detection]
            unmatched_detections_array = unmatched_detections_array[u_detection]

        """ 4.2: Create new tracks"""
        for i, inew in enumerate(unmatched_detections_array):
            bbox = inew[:4]
            score = inew[4]
            track = STrack(STrack.tlbr_to_tlwh(bbox), score)
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            if self.reid_enabled:
                if track.score > self.track_activation_threshold:  # Avoids creating embeddings for low confidence tracks
                    track.features = self.initiate_track_reid_features(track)
            self._detection_to_track_map[unmatched_detections_mask[i]] = track.track_id
            n_tracks_new += 1

        self.step_log("step 4.2", None, None, None, None)

        """ Step 5: Update state for lost tracks"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.lost_track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
            elif track.state == TrackState.Lost:
                lost_stracks.append(track)

        self.step_log("step 5", None, None, None, None)

        # Update tracker state
        if self.reid_enabled and len(scavenged_indices) > 0:
            self.removed_stracks = [t for i, t in enumerate(self.removed_stracks) if i not in scavenged_indices]
        self.removed_stracks = joint_stracks(self.removed_stracks, removed_stracks)
        
        self.tracked_stracks = joint_stracks(activated_stracks, refind_stracks)
        self.lost_stracks = lost_stracks
        self.ephemeral_stracks = joint_stracks(self.ephemeral_stracks, ephemeral_stracks)

        
        logger.debug(
            f"tracks reconfirmed: {n_tracks_reconfirmed} | "
            f"refound: {len(refind_stracks)} | "
            f"confirmed: {n_tracks_confirmed} | "
            f"new: {n_tracks_new} | "
            f"lost: {len(lost_stracks)} | "
            f"pending conf: {n_tracks_pending_confirmation} | "
            f"removed: {len(removed_stracks)} | "
            f"scavenged: {len(scavenged_indices)}"
        )
        
        if logger.getEffectiveLevel() == logging.DEBUG:
            self.log_frame_summary()

        # Return only confirmed tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks

    def log_frame_summary(self):
        """
        Log a summary of all tracks in the current frame, sorted by track ID.
        """
        all_tracks = self.tracked_stracks + self.lost_stracks + self.removed_stracks
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
            
            if status in statuses_for_metrics:
                pred_tlbr = t.predicted_tlbr
                curr_tlbr = t.tlbr
                pred_str = f"[{pred_tlbr[0]:4.0f}, {pred_tlbr[1]:3.0f}, {pred_tlbr[2]:4.0f}, {pred_tlbr[3]:3.0f}]" 
                curr_str = f"[{curr_tlbr[0]:4.0f}, {curr_tlbr[1]:3.0f}, {curr_tlbr[2]:4.0f}, {curr_tlbr[3]:3.0f}]"
                cost_str = f"{t.match_cost:.4f}" if t.match_cost is not None else "N/A"
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
        self, detections_array: np.ndarray, S_array: Optional[np.ndarray], T_array: Optional[np.ndarray], frame: Optional[np.ndarray] = None
        ) -> np.ndarray:
        """
        Update the tracker with new detections and camera motion compensation, and returns the detection array with updated tracks.
        
        This method is optimized to avoid redundant matching by tracking detection-to-track assignments during the update process.
        
        Args:
            detections_array: Array of detections in the format (N, 5) where N is the number of detections.
            S_array: Array of scale transformations for camera motion compensation.
            T_array: Array of translation transformations for camera motion compensation.
            frame: Frame to extract features from, required if reid_enabled is True.
        
        Returns:
            np.ndarray: Array with dimensions (N, 6) with updated tracks, with the format [x1, y1, x2, y2, score, track_id].
            Unknown detections are returned with track_id = -1.
        """
        original_detections = detections_array.copy()
        # Store a mapping that will be populated during update
        self._detection_to_track_map = {}
        if self.reid_enabled and frame is None:
            raise ValueError("Frame must be provided if reid_enabled is True.")
        else:
            self._frame = frame
        
        # Call update (which will populate the map through internal tracking)
        _ = self.update(detections_array, S_array, T_array)
        
        # Create output array with track IDs initialized to -1
        detections_with_tracks = np.column_stack((
            original_detections, 
            np.full(len(original_detections), -1, dtype=np.int32)
        ))
        
        # Apply the mapping
        for det_idx, track_id in self._detection_to_track_map.items():
            if det_idx < len(detections_with_tracks):
                detections_with_tracks[det_idx, 5] = track_id
        
        # Clean up temporary mapping
        delattr(self, '_detection_to_track_map')
        
        return detections_with_tracks

    def initiate_track_reid_features(self, track: STrack):
        """
        Uses class reid model to extract features from track bounding box
        """
        feats = self.get_embeddings(track.tlbr)
        if feats is not None:
            track.features = feats

    def update_track_reid_features(self, track: STrack):
        """
        Updates the track's reid features using the given detection bounding box.
        Updates every 30 frames.
        """
        if track.score < self.track_activation_threshold:
            return
        
        if track.features is None:
            self.initiate_track_reid_features(track)
            return
        
        if track.tracklet_len % 30 != 0:
            return
       
        alpha = REID_UPDATE_ALPHA
        new_feat = self.get_embeddings(track.tlbr)
        if new_feat is not None:
            updated_features = alpha * track.features + (1 - alpha) * new_feat
            track.features = torch.nn.functional.normalize(updated_features, dim=1)
            

    def get_embeddings(self, tlbr: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extracts embeddings for a given bounding box (tlbr).
        """
        if self._frame is None or self.reid_model is None:
            raise ValueError("Frame and reid_model must be set to extract embeddings.")
        
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

    def step_log(self, step, matches, unmatched_tracks, unmatched_detections, dists):
        pass
        

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

