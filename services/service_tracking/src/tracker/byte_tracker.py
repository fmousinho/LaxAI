import logging
logger = logging.getLogger(__name__)

import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from typing import Optional, List
from schemas.tracking import TrackingParams
from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState

COMPENSATE_CAM_MOTION = True  # Affine transform to track camera motion signifantly improves accuracy

LOW_CONF_MAX_MATCH_DISTANCE = 0.7
UNCONFIRMED_MAX_MATCH_DISTANCE = 0.7

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

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
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
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
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

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
    def __init__(self, args: TrackingParams, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
       
        self.track_activation_threshold = args.track_activation_threshold
        self.max_match_distance = args.max_match_distance
        self.prediction_threshold = args.prediction_threshold
        self.buffer_size = int(frame_rate / 30.0 * args.lost_track_buffer)
        self.max_time_lost = self.buffer_size
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
        
        self.frame_id += 1
        detections_array = detections_array.copy()
        S_array = S_array.copy() if S_array is not None else None
        T_array = T_array.copy() if T_array is not None else None
        
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if detections_array.shape[1] >= 5:
            scores = detections_array[:, 4]
            bboxes = detections_array[:, :4]
        else:
            detections_array = detections_array.cpu().numpy()
            scores = detections_array[:, 4] * detections_array[:, 5]
            bboxes = detections_array[:, :4]  # x1y1x2y2

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

        # Logging variables
        n_tracks_reconfirmed = 0
        n_tracks_refind = 0
        n_tracks_confirmed = 0
        n_tracks_new = 0
        n_tracks_lost = 0
        n_tracks_removed_or_unconfirmed = 0

        logger.debug(f"total detections: {len(scores_keep)} | thresh: {self.track_activation_threshold} | high conf: {len(high_conf_mask)} | low: {len(low_conf_mask)}")

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed_tracks = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_tracks.append(track)
            else:
                tracked_stracks.append(track)
        logger.debug(f"activate stracks before association: {len(tracked_stracks)} | unconfirmed: {len(unconfirmed_tracks)} | lost: {len(self.lost_stracks)}")

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Compensate the camera motion, if arrays are provided and valid
        if S_array is not None and T_array is not None and COMPENSATE_CAM_MOTION:
            # Check for NaNs or Infs which cause RuntimeWarnings
            if np.isfinite(S_array).all() and np.isfinite(T_array).all():
                STrack.multi_compensate_cam_motion(strack_pool, S_array, T_array)
            else:
                logger.warning(f"Frame {self.frame_id}: Invalid camera motion arrays (NaN/Inf detected). Skipping compensation.")

        STrack.multi_predict(strack_pool)
        strack_pool_array = np.array([track.tlbr for track in strack_pool])

        if len(strack_pool_array) > 0 and len(bboxes_high) > 0:
            dists = matching.iou_distance(strack_pool_array, bboxes_high)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.max_match_distance)
        else:
            matches = []
            u_track = np.arange(len(strack_pool_array))
            u_detection = np.arange(len(bboxes_high))
        
        unmatched_tracks_array = strack_pool_array[u_track]
        unmatched_tracks = [strack_pool[i] for i in u_track]

        for itracked, idet in matches:
            track = strack_pool[itracked]
            bbox = bboxes_high[idet]
            score = scores_high[idet]
            self._detection_to_track_map[high_conf_mask[idet]] = track.track_id
            
            if track.state == TrackState.Tracked:
                track.update(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id)
                activated_stracks.append(track)
                n_tracks_reconfirmed += 1
            else:
                track.re_activate(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id, new_id=False)
                refind_stracks.append(track)
                n_tracks_refind += 1

        ''' Step 3: Second association, with low score detection boxes'''
        
        if len(unmatched_tracks_array) > 0 and len(bboxes_low) > 0:
            dists = matching.iou_distance(unmatched_tracks_array, bboxes_low)
            matches, u_track_second, u_detection_second = matching.linear_assignment(dists, thresh=LOW_CONF_MAX_MATCH_DISTANCE)
        else:
            matches = []
            u_track_second = np.arange(len(unmatched_tracks_array))
            u_detection_second = np.arange(len(bboxes_low))

        unmatched_tracks_second = [unmatched_tracks[i] for i in u_track_second]
        unmatched_tracks_array_second = unmatched_tracks_array[u_track_second]

        for itracked, idet in matches:
            track = unmatched_tracks[itracked]
            bbox = bboxes_low[idet]
            score = scores_low[idet]
            self._detection_to_track_map[low_conf_mask[idet]] = track.track_id

            if track.state == TrackState.Tracked:
                track.update(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id)
                activated_stracks.append(track)
                n_tracks_reconfirmed += 1
            else:
                track.re_activate(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id, new_id=False)
                refind_stracks.append(track)
                n_tracks_refind += 1

        for track in unmatched_tracks_second:
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                n_tracks_lost += 1
                logger.debug(f"lost track: {track.track_id}  |  bbox: {track.tlbr.astype(int)} | score: {track.score:.2f}")

        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        unmatched_detections_mask = np.concatenate((high_conf_mask[u_detection], low_conf_mask[u_detection_second]), axis=0)
        unmatched_bbox_array = bboxes_keep[unmatched_detections_mask]
        unmatched_scores_array = scores_keep[unmatched_detections_mask]
        unconfirmed_tracks_array = np.array([track.tlbr for track in unconfirmed_tracks]) if len(unconfirmed_tracks) > 0 else np.empty((0, 4))

        if logger.level == logging.DEBUG:
            for i in unmatched_detections_mask:
                logger.debug(f"unmatched detection: {bboxes_keep[i].astype(int)} | score: {scores_keep[i]:.2f}")

        if len(unconfirmed_tracks_array) > 0 and len(unmatched_bbox_array) > 0:
            dists = matching.iou_distance(unconfirmed_tracks_array, unmatched_bbox_array)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=UNCONFIRMED_MAX_MATCH_DISTANCE)
        else:
            matches = []
            u_unconfirmed = np.arange(len(unconfirmed_tracks))
            u_detection = np.arange(len(unmatched_bbox_array))

        for itracked, idet in matches:
            track = unconfirmed_tracks[itracked]
            bbox = unmatched_bbox_array[idet]
            score = unmatched_scores_array[idet]
            track.update(STrack(STrack.tlbr_to_tlwh(bbox), score), self.frame_id)
            activated_stracks.append(track)
            self._detection_to_track_map[unmatched_detections_mask[idet]] = track.track_id
            n_tracks_removed_or_unconfirmed += 1

        unmatched_detections_mask = unmatched_detections_mask[u_detection]
        unmatched_detections_array = np.column_stack((bboxes_keep[unmatched_detections_mask], scores_keep[unmatched_detections_mask])) if len(unmatched_detections_mask) > 0 else np.empty((0, 5))
        unconfirmed_tracks_remaining = [unconfirmed_tracks[i] for i in u_unconfirmed]

        for track in unconfirmed_tracks_remaining:
            track.mark_removed()
            removed_stracks.append(track)
            n_tracks_removed_or_unconfirmed += 1

        """ Step 4: Init new stracks"""
        for i, inew in enumerate(unmatched_detections_array):
            bbox = inew[:4]
            score = inew[4]
            track = STrack(STrack.tlbr_to_tlwh(bbox), score)
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            self._detection_to_track_map[unmatched_detections_mask[i]] = track.track_id
            n_tracks_new += 1

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                n_tracks_removed_or_unconfirmed += 1

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug (
            f"tracks reconfirmed: {n_tracks_reconfirmed} | "
            f"refound: {n_tracks_refind} | "
            f"confirmed: {n_tracks_confirmed} | "
            f"new: {n_tracks_new} | "
            f"lost: {n_tracks_lost} | "
            f"removed/unconfirmed: {n_tracks_removed_or_unconfirmed}"
            )
        return output_stracks


    def assign_tracks_to_detections(self, detections_array: np.ndarray, S_array: Optional[np.ndarray], T_array: Optional[np.ndarray]) -> np.ndarray:
        """
        Update the tracker with new detections and camera motion compensation, and returns the detection array with updated tracks.
        
        This method is optimized to avoid redundant matching by tracking detection-to-track assignments during the update process.
        
        Args:
            detections_array: Array of detections in the format (N, 5) where N is the number of detections.
            S_array: Array of scale transformations for camera motion compensation.
            T_array: Array of translation transformations for camera motion compensation.
        
        Returns:
            np.ndarray: Array with dimensions (N, 6) with updated tracks, with the format [x1, y1, x2, y2, score, track_id].
            Unknown detections are returned with track_id = -1.
        """
        original_detections = detections_array.copy()
        # Store a mapping that will be populated during update
        self._detection_to_track_map = {}
        
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

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb