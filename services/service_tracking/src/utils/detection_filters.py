"""
Utility functions for detection post-processing.
"""
import numpy as np


def nms(detections: np.ndarray, iou_threshold: float = 0.4) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to remove duplicate/overlapping detections.
    
    Args:
        detections: np.ndarray of shape (N, 5) with format [x1, y1, x2, y2, confidence]
        iou_threshold: IoU threshold for considering boxes as duplicates
        
    Returns:
        np.ndarray: Filtered detections with duplicates removed
    """
    if len(detections) == 0:
        return detections
    
    # Extract coordinates and scores
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
    
    # Compute areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # Pick the detection with highest score
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # Compute IoU with remaining detections
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep only detections with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return detections[keep]


def reject_ambiguous_detections(detections: np.ndarray, track_predictions: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """
    Reject detections that significantly overlap with multiple track predictions.
    This helps prevent incorrect assignments when multiple tracks compete for the same detection.
    
    Args:
        detections: np.ndarray of shape (N, 5+) with format [x1, y1, x2, y2, confidence, ...]
        track_predictions: np.ndarray of shape (M, 4) with predicted track bboxes in tlbr format
        iou_threshold: IoU threshold for considering overlap as "ambiguous"
        
    Returns:
        np.ndarray: Filtered detections with ambiguous ones removed
    """
    if len(detections) == 0 or len(track_predictions) == 0:
        return detections
    
    keep_mask = np.ones(len(detections), dtype=bool)
    
    for i, det in enumerate(detections):
        det_box = det[:4]
        
        # Count how many tracks this detection overlaps with
        overlap_count = 0
        for track_box in track_predictions:
            iou = compute_iou(det_box, track_box)
            if iou > iou_threshold:
                overlap_count += 1
        
        # If detection overlaps with 2+ tracks, it's ambiguous - reject it
        if overlap_count >= 2:
            keep_mask[i] = False
    
    return detections[keep_mask]


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in tlbr format.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def remove_border_detections(detections: np.ndarray, frame_size: tuple[int, int] = (1920, 1080), margin: int = 2) -> np.ndarray:
    """
    Removes detections that touch the border to improve embedding quality.
    
    Args:
        detections: np.ndarray of shape (N, 5+) with format [x1, y1, x2, y2, confidence, ...]
        frame_size: (width, height) of the frame
        margin: pixel margin from border
        
    Returns:
        np.ndarray: Detections with border-touching ones removed
    """
    if len(detections) == 0:
        return detections
    
    frame_width, frame_height = frame_size
    
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    
    # Keep detections that don't touch borders
    keep_mask = (
        (x1 >= margin) &
        (y1 >= margin) &
        (x2 <= frame_width - margin) &
        (y2 <= frame_height - margin)
    )
    
    return detections[keep_mask]


def filter_detections_pipeline(
    detections: np.ndarray,
    frame_size: tuple[int, int],
    track_predictions: np.ndarray | None = None,
    nms_iou_threshold: float = 0.3,
    border_margin: int = 2,
    ambiguous_iou_threshold: float = 0.4,
) -> np.ndarray:
    """
    Apply all pre-tracking detection filters in sequence.
    
    Pipeline:
    1. NMS (remove duplicate detections)
    2. Border removal (remove detections touching frame edges)  
    3. Ambiguous rejection (remove detections overlapping with 2+ tracks, if tracks provided)
    
    Args:
        detections: Raw detections from detector [x1, y1, x2, y2, confidence, ...]
        frame_size: (width, height) of the frame
        track_predictions: Optional array of track predictions for ambiguous rejection
        nms_iou_threshold: IoU threshold for NMS
        border_margin: Pixel margin from border
        ambiguous_iou_threshold: IoU threshold for ambiguous rejection
        
    Returns:
        np.ndarray: Filtered detections ready for tracking
    """
    if len(detections) == 0:
        return detections
    
    # Step 1: NMS - remove duplicates
    detections = nms(detections, iou_threshold=nms_iou_threshold)
    
    # Step 2: Border removal
    detections = remove_border_detections(detections, frame_size, margin=border_margin)
    
    # Step 3: Ambiguous rejection (only if tracks provided)
    if track_predictions is not None and len(track_predictions) > 0 and len(detections) > 0:
        detections = reject_ambiguous_detections(detections, track_predictions, iou_threshold=ambiguous_iou_threshold)
    
    return detections
