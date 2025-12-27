import logging
logger = logging.getLogger(__name__)
from typing import List, Callable, Tuple, Optional
import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        cost_matrix: cost matrix
        thresh: maximum cost (distance) for a match to be considered
    Returns:
        matches: array with indices of matches
        unmatched_a: array with indices of unmatched detections
        unmatched_b: array with indices of unmatched tracks
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def aspect_ratio_distance(tracks: List, detections: np.ndarray) -> np.ndarray:
    """
    Compute cost based on aspect ratio difference
    :type tracks: list[STrack]
    :type detections: np.ndarray
    :rtype cost_matrix np.ndarray
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))

    tracks_xyah = np.array([track.to_xyah() for track in tracks])
    detections_tlbr = detections[:, :4]

    tracks_ar = tracks_xyah[:, 2]

    detections_w = detections_tlbr[:, 2] - detections_tlbr[:, 0]
    detections_h = detections_tlbr[:, 3] - detections_tlbr[:, 1]
    detections_ar = detections_w / (detections_h + 1e-6)

    tracks_ar = tracks_ar[:, np.newaxis] # (N, 1)
    detections_ar = detections_ar[np.newaxis, :] # (1, M)

    r = abs(np.log(tracks_ar) - np.log(detections_ar))
    
    cost_matrix = r
    
    return cost_matrix

def scale_distance(tracks: List, detections: np.ndarray) -> np.ndarray:
    """
    Compute cost based on scale difference
    :type tracks: list[STrack]
    :type detections: np.ndarray
    :rtype cost_matrix np.ndarray
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))
    
    tracks_xyah = np.array([track.to_xyah() for track in tracks])
    detections_tlbr = detections[:, :4]

    tracks_h = tracks_xyah[:, 3]
    detections_h = detections_tlbr[:, 3] - detections_tlbr[:, 1]
    
    tracks_h = tracks_h[:, np.newaxis] # (N, 1)
    detections_h = detections_h[np.newaxis, :] # (1, M)
    
    r1 = tracks_h / (detections_h + 1e-6)
    r2 = detections_h / (tracks_h + 1e-6)
    
    min_r = np.minimum(r1, r2)
    cost_matrix = 1 - min_r
    
    return cost_matrix


def v_iou_distance(tracks: List, detections_arr: np.ndarray) -> np.ndarray:
    """
    Compute cost based on IoU
    :type tracks: list[STrack]
    :type detections_arr: np.ndarray in tlbr format

    :rtype cost_matrix: np.ndarray
    """

    tracks_tlbr = np.array([track.tlbr for track in tracks])
    detections_tlbr = detections_arr[:, :4]
    if tracks_tlbr.shape[0] == 0 or detections_tlbr.shape[0] == 0:
        cost_matrix = np.empty((tracks_tlbr.shape[0], detections_tlbr.shape[0]))
    else:
        _ious = ious(tracks_tlbr, detections_tlbr)
        cost_matrix = 1 - _ious

    return cost_matrix

def spatial_distance(tracks: List, detections_tlbr: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean distance between track centers and detection centers.
    
    :param tracks: list[STrack] - tracks with .tlbr attribute
    :param detections_tlbr: np.ndarray - detection bboxes (N, 4) in tlbr format
    :return: distance matrix of shape (len(tracks), len(detections))
    """
    if len(tracks) == 0 or len(detections_tlbr) == 0:
        return np.empty((len(tracks), len(detections_tlbr)))
    
    # Track centers from tlbr
    track_centers = np.array([
        [(t.tlbr[0] + t.tlbr[2]) / 2, (t.tlbr[1] + t.tlbr[3]) / 2]
        for t in tracks
    ])
    # Detection centers from tlbr
    det_centers = np.column_stack([
        (detections_tlbr[:, 0] + detections_tlbr[:, 2]) / 2,
        (detections_tlbr[:, 1] + detections_tlbr[:, 3]) / 2
    ])
    return cdist(track_centers, det_centers, metric='euclidean')

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.features for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    :param kf: KalmanFilter object
    :param cost_matrix: np.ndarray
    :param tracks: list[STrack]
    :param detections: np.ndarray in tlbr format
    :param only_position: bool
    :return: cost_matrix np.ndarray
    """
    detections_xyah = np.zeros_like(detections)
    detections_xyah[:, 0] = (detections[:, 0] + detections[:, 2]) / 2 # x
    detections_xyah[:, 1] = (detections[:, 1] + detections[:, 3]) / 2 # y
    detections_xyah[:, 3] = detections[:, 3] - detections[:, 1]       # h
    # Avoid division by zero
    detections_xyah[:, 2] = (detections[:, 2] - detections[:, 0]) / (detections_xyah[:, 3] + 1e-6) # a
    
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, detections_xyah, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def v_iou_reid_distance(
    tracks: List, 
    detections_bboxes: np.ndarray, 
    embedding_func: Callable, 
    kalman_filter_obj: Optional[kalman_filter.KalmanFilter] = None,
    iou_thresh=0.0, 
    reid_weight=0.7,
) -> np.ndarray:
    """
    Computes a cost matrix blending IoU and ReID distance.
    Only computes ReID distance for pairs with IoU > iou_thresh.
    
    :param tracks: list[STrack]
    :param detections_bboxes: np.ndarray in tlbr format
    :param embedding_func: callable(tlbr) -> tensor/numpy
    :param kalman_filter_obj: KalmanFilter object
    :param iou_thresh: lower bound for overlap to consider ReID
    :param reid_weight: weight for ReID distance (0.0 = only IoU, 1.0 = only ReID)
    :return: cost_matrix np.ndarray
    """
    if len(tracks) == 0 or len(detections_bboxes) == 0:
        return np.empty((len(tracks), len(detections_bboxes)))

    if detections_bboxes.shape[1] > 4:
        detections_bboxes = detections_bboxes[:, :4]
    elif detections_bboxes.shape[1] < 4:
        raise ValueError("detections_bboxes must be of shape (N, 4)")
        
    # 1. Compute IoU Cost Matrix (1 - IoU)
    cost_matrix = v_iou_distance(tracks, detections_bboxes)
    
    # 2. Apply spatial gating if configured
    if kalman_filter_obj is not None:
        cost_matrix = gate_cost_matrix(kalman_filter_obj, cost_matrix, tracks, detections_bboxes)
    
    # Identify candidates where we have some overlap
    # We want similarity = 1 - cost > thresh => cost < 1 - thresh
    # Or just check simple IoU from 1-cost
    ious = 1.0 - cost_matrix
    candidates_mask = ious >= iou_thresh
    
    if not np.any(candidates_mask):
        return cost_matrix
        
    # Cache for detection embeddings to avoid re-running model for same detection
    det_embeddings_cache = {}
    
    # 3. Iterate and apply ReID
    # We'll modify the cost matrix in place for candidates
    rows, cols = np.where(candidates_mask)
    
    for r, c in zip(rows, cols):
        track = tracks[r]
        det = detections_bboxes[c]
        
        # Skip if already gated
        if cost_matrix[r, c] == np.inf:
            continue
        
        # Get Track Embedding
        # Assuming track has .features attribute which is a tensor or numpy array
        if track.features is None:
            continue
            
        track_feat = track.features
        if hasattr(track_feat, 'cpu'):
            track_feat = track_feat.cpu().numpy()
        if len(track_feat.shape) == 1:
            track_feat = track_feat.reshape(1, -1)
            
        # Get Detection Embedding
        if c not in det_embeddings_cache:
            # Assuming det is an STrack or has tlbr property
            # Or det is the STrack associated with detection
            bbox = det
            feat = embedding_func(bbox)
            if feat is None:
                det_embeddings_cache[c] = None
                continue
                
            if hasattr(feat, 'cpu'):
                feat = feat.cpu().numpy()
            if len(feat.shape) == 1:
                feat = feat.reshape(1, -1)
            det_embeddings_cache[c] = feat
            
        det_feat = det_embeddings_cache[c]
        
        if det_feat is None:
            continue
            
        # Compute Cosine Distance (1 - Cosine Similarity)
        # cdist returns distance. 'cosine' is 1 - u.v/(|u||v|)
        # Range is [0, 2]. Normalize to [0, 1] by dividing by 2.0
        reid_dist = cdist(track_feat, det_feat, metric='cosine')[0][0] / 2.0
        
        # 4. Fuse Scores
        # Weighted average of the IoU distance and ReID distance
        iou_d = cost_matrix[r, c]
        
        # Blend
        fused_cost = (1 - reid_weight) * iou_d + reid_weight * reid_dist
        
        cost_matrix[r, c] = fused_cost

        if logger.getEffectiveLevel() == logging.DEBUG:
            if iou_d < 1 and reid_dist < 1:
                logger.debug(f"Track {track.track_id} vs Detection {det.astype(int)}: IoU dist {iou_d:.4f}, ReID dist {reid_dist:.4f}, Fused dist {fused_cost:.4f}")
        
    return cost_matrix

# ===========================================================================
# Cost Matrix Modifiers (Penalties & Gates)
# ===========================================================================

def penalize_aspect_ratio(cost_matrix, tracks, detections, factor=0.5):
    """Add penalty to cost matrix based on aspect ratio changes."""
    if len(tracks) != cost_matrix.shape[0] or len(detections) != cost_matrix.shape[1]:
        raise ValueError("Cost matrix dimensions do not match")
    cost_matrix += aspect_ratio_distance(tracks, detections) * factor
    return cost_matrix


def penalize_scale(cost_matrix, tracks, detections, factor=1.0):
    """Add penalty to cost matrix based on scale changes."""
    if len(tracks) != cost_matrix.shape[0] or len(detections) != cost_matrix.shape[1]:
        raise ValueError("Cost matrix dimensions do not match")
    cost_matrix += scale_distance(tracks, detections) * factor
    return cost_matrix


def gate_height(cost_matrix, tracks, detections, threshold=0.2):
    """Gate association if height change exceeds threshold."""
    if cost_matrix.size == 0 or len(tracks) == 0 or len(detections) == 0:
        return cost_matrix
    
    det_heights = detections[:, 3] - detections[:, 1]
    track_heights = np.array([(t.tlbr[3] - t.tlbr[1]) for t in tracks])
    
    t_h = track_heights[:, np.newaxis]
    d_h = det_heights[np.newaxis, :]
    diff = np.abs(t_h - d_h) / (t_h + 1e-6)
    
    cost_matrix[diff > threshold] = np.inf
    return cost_matrix


def enforce_min_distance(cost_matrix, tracks, detections, min_dist: float = .2):
    """
    Gate association if best cost is not sufficiently better than second-best.
    
    For each track (row), if the difference between the best and second-best detection
    is less than min_dist, set all costs for that track to infinity (no valid match).
    
    Args:
        cost_matrix: Current cost matrix (tracks x detections)
        tracks: List of tracks
        detections: Detection array
        min_dist: Minimum required difference between best and second-best cost
        
    Returns:
        Modified cost matrix with ambiguous matches gated
    """
    if cost_matrix.size == 0 or len(tracks) == 0 or len(detections) == 0:
        return cost_matrix
    
    # For each track (row), check if best match is sufficiently better than second-best
    for track_idx in range(cost_matrix.shape[0]):
        row_costs = cost_matrix[track_idx, :]
        
        # Get valid (non-infinite) costs
        valid_costs = row_costs[row_costs < np.inf]
        
        if len(valid_costs) >= 2:
            # Sort to get best and second-best
            sorted_costs = np.sort(valid_costs)
            best_cost = sorted_costs[0]
            second_best_cost = sorted_costs[1]
            
            # If margin is too small, invalidate all matches for this track
            if (second_best_cost - best_cost) < min_dist:
                cost_matrix[track_idx, :] = np.inf
    
    return cost_matrix
    


def compute_association_cost(
    tracks,
    detections,
    kalman_filter_obj=None,
    apply_aspect_ratio_penalty=True,
    apply_height_gate=True,
    apply_enforce_min_distance=True,
    aspect_ratio_factor=0.5,
    height_threshold=0.2,
    enforce_min_distance_threshold=0.1,
):
    """Compute full association cost matrix with all penalties and gates applied."""
    bboxes = detections[:, :4]
    cost_matrix = v_iou_distance(tracks, bboxes)
    
    if apply_aspect_ratio_penalty:
        cost_matrix = penalize_aspect_ratio(cost_matrix, tracks, detections, factor=aspect_ratio_factor)
    
    if apply_height_gate:
        cost_matrix = gate_height(cost_matrix, tracks, detections, threshold=height_threshold)
    
    if kalman_filter_obj is not None:
        cost_matrix = gate_cost_matrix(kalman_filter_obj, cost_matrix, tracks, detections)
    
    if apply_enforce_min_distance:
        cost_matrix = enforce_min_distance(cost_matrix, tracks, detections)
    
    return cost_matrix
