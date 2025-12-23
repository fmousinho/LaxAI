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
from .kalman_filter import KalmanFilter, detection_np_tlbr_to_xyah
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
        self.initial_tlwh = np.asarray(tlwh, dtype=float)

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
        self.tracklet_len = 0
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
        return self.initial_tlwh.copy()


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

        detections = remove_border_detections_from(detections_array, frame_size=self.frame_size)

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
            strack_pool, detections_high, max_match_distance=0.6
        )
        activated_stracks.extend(newly_activated_stracks)
        refind_stracks.extend(newly_refind_stracks)
        pending_stracks.extend(newly_pending_stracks)

        ''' Step 2: Second association, with low score detection boxes'''
        logger.debug(f"=== Step 2: Association with low score detections ====")
        (newly_activated_stracks, newly_refind_stracks, newly_pending_stracks, unmatched_stracks, unmatched_detections_s2) = self.association_step(
            unmatched_stracks, detections_low, max_match_distance=0.8
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
        #logger.debug(f"=== Step 4: Create new tracks ========================")
        for detection in unmatched_detections:
            bbox = bboxes_from(detection[np.newaxis, :])[0]
            score = scores_from(detection[np.newaxis, :])[0]
            if score < self.prediction_threshold:
                continue
            track = STrack(STrack.tlbr_to_tlwh(bbox), score)
            track.activate(self.kalman_filter, self.frame_id)
            pending_stracks.append(track)
            self.update_with_track(detection, track)

        """ Step 5: Update state for lost tracks"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.lost_track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
            elif track.state == TrackState.Lost:
                lost_stracks.append(track)

        self.tracked_stracks = activated_stracks + refind_stracks
        self.lost_stracks = lost_stracks
        self.pending_stracks = pending_stracks
        self.removed_stracks.extend(removed_stracks)
        self.ephemeral_stracks.extend(ephemeral_stracks)
        
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
            cost_matrix = matching.v_iou_distance(strack_pool, bboxes)
            panalize_aspect_ratio_swings(cost_matrix, strack_pool, bboxes)
            penalize_scale_swings(cost_matrix, strack_pool, bboxes)
            matches, u_track, u_detection = matching.linear_assignment(cost_matrix, thresh=max_match_distance)
            
            for itracked, idet in matches:
                strack = strack_pool[itracked]
                bbox = bboxes[idet]
                score = scores[idet]
                strack.match_cost = cost_matrix[itracked, idet]
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

                logger.debug(f"Track {strack.track_id:4d} matched with detection {idet:2d} with cost {cost_matrix[itracked, idet]:.4f}")

            unmatched_tracks = [strack_pool[i] for i in u_track]
            if logger.getEffectiveLevel() == logging.DEBUG:
                for i in u_track:
                    logger.debug(f"Track {strack_pool[i].track_id:4d} unmatched, best cost {cost_matrix[i].min():.4f}") 
            
            unmatched_detections = detections[u_detection]
            if logger.getEffectiveLevel() == logging.DEBUG:
                for i in u_detection:
                    logger.debug(f"Detection {i:2d} unmatched, best cost {cost_matrix[:, i].min():.4f}")

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
        # Store reference for update_with_track
        self._original_detections = detections
        
        _ = self.update(detections, S_array, T_array)

        confirmed_tracks_only = True
        self._original_detections[:, 5] *= self._original_detections[:, 7] if confirmed_tracks_only else 1 
 
        return self._original_detections[:, :6]


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

def remove_border_detections_from(detections: np.ndarray, frame_size: Tuple[int, int] = (1920, 1080), margin: int = 10) -> np.ndarray:
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

def panalize_aspect_ratio_swings(cost_matrix: np.ndarray, stracks: List[STrack], detections: np.ndarray, factor: float = 1.0):
    if len(stracks) != cost_matrix.shape[0] or len(detections) != cost_matrix.shape[1]:
        raise ValueError("Cost matrix dimensions do not match the number of tracks and detections")
    cost_matrix += matching.aspect_ratio_distance(stracks, detections) * factor

def penalize_scale_swings(cost_matrix: np.ndarray, stracks: List[STrack], detections: np.ndarray, factor: float = 1.2):
    if len(stracks) != cost_matrix.shape[0] or len(detections) != cost_matrix.shape[1]:
        raise ValueError("Cost matrix dimensions do not match the number of tracks and detections")
    cost_matrix += matching.scale_distance(stracks, detections) * factor

def bboxes_from(detections: np.ndarray) -> np.ndarray:
    return detections[:, :4]
        
def scores_from(detections: np.ndarray) -> np.ndarray:
    return detections[:, 4]

