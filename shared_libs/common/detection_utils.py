"""
Detection utilities for the LaxAI project.

This module provides functions for running detection and tracking on video frames,
including integration with detection models and trackers, and saving results.
"""

import json
import logging
import os
from collections import defaultdict
from typing import List, Optional, Dict, Any

import numpy as np
from supervision import (Detections, JSONSink, VideoInfo,
                         get_video_frames_generator)

from shared_libs.common.detection import DetectionModel
from shared_libs.common.tracker import AffineAwareByteTrack
from shared_libs.config.all_config import detection_config

logger = logging.getLogger(__name__)


def _detections_to_summary_dict(merged: Detections, detections_list: List[Detections], extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Internal helper to build a canonical summary JSON dict for a merged Detections object.

    The summary includes coordinate arrays plus optional frame_index (if present in data/metadata),
    and total counts. This unifies previous ad-hoc dict constructions across pipelines.
    """
    # Frame indices: attempt to gather from each element's data['frame_index'] else metadata['frame_id']
    frame_indices: List[int] = []
    collected = False
    try:
        # If merged already has frame_index in data (preferred)
        if merged.data and 'frame_index' in merged.data:
            fi = merged.data.get('frame_index')
            if isinstance(fi, (list, np.ndarray)):
                frame_indices = list(fi)
                collected = True
    except Exception:  # pragma: no cover - defensive
        pass

    if not collected:
        # Build by iterating original detections objects
        for det in detections_list:
            per_rows = len(det)
            if per_rows == 0:
                continue
            # Priority 1: explicit per-row frame_index list
            if det.data and 'frame_index' in det.data and isinstance(det.data['frame_index'], (list, np.ndarray)):
                fi_list = det.data['frame_index']
                if len(fi_list) == per_rows:
                    frame_indices.extend([int(x) if x is not None else -1 for x in fi_list])
                    continue
            # Priority 2: single frame_id metadata replicated
            frame_id_meta = None
            if hasattr(det, 'metadata') and isinstance(det.metadata, dict):
                frame_id_meta = det.metadata.get('frame_id')
            if frame_id_meta is not None:
                frame_indices.extend([int(frame_id_meta)] * per_rows)
            else:
                # Unknown frame indices: placeholder -1
                frame_indices.extend([-1] * per_rows)

    # Post-process: if any -1 or None placeholders remain, attempt second pass using metadata
    if any(fi is None or fi == -1 for fi in frame_indices):
        repaired: List[int] = []
        iter_idx = 0
        for det in detections_list:
            per_rows = len(det)
            if per_rows == 0:
                continue
            meta_frame = None
            if hasattr(det, 'metadata') and isinstance(det.metadata, dict):
                meta_frame = det.metadata.get('frame_id')
            for j in range(per_rows):
                current = frame_indices[iter_idx]
                if (current is None or current == -1) and meta_frame is not None:
                    repaired.append(int(meta_frame))
                else:
                    # If still unknown, fallback 0 to keep length consistent
                    repaired.append(0 if current is None or current == -1 else int(current))
                iter_idx += 1
        frame_indices = repaired

    tracker_id_list = []
    try:
        if getattr(merged, 'tracker_id', None) is not None:
            tracker_id_list = merged.tracker_id.tolist()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        tracker_id_list = []

    summary: Dict[str, Any] = {
        'xyxy': merged.xyxy.tolist() if merged.xyxy is not None else [],
        'confidence': merged.confidence.tolist() if merged.confidence is not None else [],
        'class_id': merged.class_id.tolist() if merged.class_id is not None else [],
        'tracker_id': tracker_id_list,
        'frame_index': frame_indices,
        'data': merged.data if getattr(merged, 'data', None) else {},
        'total_frames': len(detections_list),
        'total_detections': len(merged),
    }
    if extra_metadata:
        summary.update(extra_metadata)
    return summary


def save_all_detections(storage_client, blob_name: str, detections_list: List[Detections], extra_metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Serialize and upload a list of Detections objects as a single summary JSON.

    Args:
        storage_client: Object exposing upload_from_bytes(destination, data)
        blob_name: Target blob path (should end with .json)
        detections_list: List of sv.Detections across frames
        extra_metadata: Optional additional key-values to append to summary

    Returns:
        bool: True if upload succeeded, False otherwise
    """
    try:
        if not detections_list:
            empty_summary = {
                'xyxy': [], 'confidence': [], 'class_id': [], 'tracker_id': [],
                'frame_index': [], 'data': {}, 'total_frames': 0, 'total_detections': 0
            }
            return storage_client.upload_from_bytes(blob_name, empty_summary)

        # Before merging, some sv.Detections implementations require identical data keys across
        # all objects. Our tests deliberately remove frame_index on one object to verify fallback
        # to metadata['frame_id']. We build a normalized list where each detection has the union
        # of keys, filling missing keys with appropriate length placeholders.
        try:
            # Determine union of data keys (excluding metadata-only keys)
            union_keys = set()
            per_lengths: List[int] = []
            for det in detections_list:
                per_lengths.append(len(det))
                if getattr(det, 'data', None):
                    union_keys.update(det.data.keys())
            # Ensure frame_index is preserved if present anywhere or if metadata will supply it
            if 'frame_index' not in union_keys:
                # We'll add it later from metadata fallback if needed
                pass
            normalized: List[Detections] = []
            for det, row_len in zip(detections_list, per_lengths):
                data_dict: Dict[str, Any] = {}
                source_data = getattr(det, 'data', {}) or {}
                for k in union_keys:
                    if k in source_data:
                        data_dict[k] = source_data[k]
                    else:
                        # Fill missing with list of Nones so lengths align; downstream consumers
                        # ignore unknown data fields or Nones.
                        data_dict[k] = [None] * row_len
                # We'll handle frame_index separately after merging; remove conflicting keys
                normalized.append(Detections(
                    xyxy=det.xyxy,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    tracker_id=getattr(det, 'tracker_id', None),
                    data=data_dict
                ))
            merged = Detections.merge(normalized)
        except Exception as merge_err:  # pragma: no cover
            logger.error(f"Normalization merge failed, fallback to naive merge attempt: {merge_err}")
            merged = Detections.merge(detections_list)

        summary = _detections_to_summary_dict(merged, detections_list, extra_metadata)
        return storage_client.upload_from_bytes(blob_name, summary)
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"Failed to save all detections to {blob_name}: {e}")
        return False


def load_all_detections_summary(summary_json: Dict[str, Any]) -> Detections:
    """Load a merged summary JSON (created by save_all_detections) back into a Detections object.

    Also handles the serialized format where 'all_detections' contains serialized detection strings.

    Note: frame_index is preserved in the returned object's data as 'frame_index'.

    Args:
        summary_json: Parsed JSON dict from detections summary file

    Returns:
        Detections: Single merged detection object representing all rows.
    """
    try:
        # Check if this is the serialized format (has 'all_detections' with strings)
        if 'all_detections' in summary_json and isinstance(summary_json['all_detections'], list):
            # This is the serialized format - delegate to the specialized loader
            return load_serialized_detections(summary_json)

        # Original summary format handling
        xyxy = np.array(summary_json.get('xyxy', []), dtype=np.float32) if summary_json.get('xyxy') else np.empty((0,4), dtype=np.float32)
        class_id_raw = summary_json.get('class_id', [])
        class_id = np.array(class_id_raw, dtype=int) if class_id_raw else None
        confidence_raw = summary_json.get('confidence', [])
        confidence = np.array(confidence_raw, dtype=np.float32) if confidence_raw else None
        tracker_id_raw = summary_json.get('tracker_id', [])
        tracker_id = np.array(tracker_id_raw, dtype=int) if tracker_id_raw else None
        data_field = summary_json.get('data', {}) or {}
        frame_index = summary_json.get('frame_index', [])
        if frame_index:
            data_field = dict(data_field)  # shallow copy
            data_field['frame_index'] = np.array(frame_index, dtype=int)
        # Ensure data_field is a dict[str, list|ndarray]
        if not isinstance(data_field, dict):
            data_field = {}
        # Cast to expected type to satisfy type checker
        from typing import cast, Dict, Union, List
        data_field = cast(Dict[str, Union[np.ndarray, List]], data_field)
        det = Detections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            tracker_id=tracker_id,
            data=data_field
        )
        return det
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to load detections summary: {e}")
        return Detections.empty()


def process_frames(
    model: DetectionModel,
    tracker: AffineAwareByteTrack,
    frames_generator,
    frame_target: int,
    detection_file_path: str,
    nms_iou: Optional[float] = None,
) -> List[Detections]:
    """
    Process all frames for detection and tracking.

    Args:
        model: Detection model instance
        tracker: Tracker instance
        frames_generator: Generator yielding video frames
        frame_target: Total number of frames to process
        detection_file_path: Path to save the detections JSON file
        nms_iou: NMS IoU threshold (if None, uses config default)

    Returns:
        List of sv.Detections objects with loaded detections
    """
    if nms_iou is None:
        nms_iou = detection_config.nms_iou_threshold

    if nms_iou is not None:
        logger.info(f"Using NMS IoU threshold: {nms_iou}")

    logger.info("Generating detections and tracks for each frame")

    multi_frame_detections = list()
    previous_frame = None
    affine_matrix = None
    current_frame_idx = 0

    json_sink = JSONSink(detection_file_path)

    with json_sink as sink:
        for frame in frames_generator:
            if current_frame_idx >= frame_target:
                break

        # Generate detections for current frame
        frame_detections = model.generate_detections(frame)

        if len(frame_detections) == 0:
            # If no detections, create an empty object for len to match frame count
            frame_detections = model.empty_detections()
        else:
            # Ensure we have a single Detections object (not a list)
            if isinstance(frame_detections, list):
                if len(frame_detections) > 0:
                    frame_detections = frame_detections[0]  # Take first detection if it's a list
                else:
                    frame_detections = model.empty_detections()
            
            if frame_detections.xyxy.size > 0 and np.any(frame_detections.xyxy < 0):
                logger.debug(f"Negative coordinates for frame {current_frame_idx}.")

        # At this point, frame_detections should always be a single Detections object
        assert not isinstance(frame_detections, list), "frame_detections should be a single Detections object"

        if previous_frame is not None:
            affine_matrix = AffineAwareByteTrack.calculate_affine_transform(previous_frame, frame)
        if affine_matrix is None:
            affine_matrix = AffineAwareByteTrack.get_identity_affine_matrix()

        frame_detections.metadata.update({"frame_id": current_frame_idx})

        if nms_iou is not None and len(frame_detections) > 1:
            frame_detections = frame_detections.with_nms(threshold=nms_iou)

        detections = tracker.update_with_transform(
            detections=frame_detections, frame=frame, affine_matrix=affine_matrix
        )

        # Store detections and update state
        multi_frame_detections.append(detections)
        previous_frame = frame.copy()
        current_frame_idx += 1

        # Write detections to JSON file
        sink.append(detections, custom_data={"frame_id": current_frame_idx})

        # Log progress
        logger.info(f"Frames read: {current_frame_idx}/{frame_target}")

    logger.info("Detection and tracking processing complete.")
    logger.info(f"JSON saved to: {detection_file_path}")
    logger.info(f"Found {tracker.get_n_of_tracks()} total tracks.")

    return multi_frame_detections


def load_detections_from_json(
    model: DetectionModel,
    tracker: AffineAwareByteTrack,
    json_file_path: str,
    video_source: str,
    update_tracker_state: bool = True,
) -> List[Detections]:
    """
    Load detections from a JSON file, and optionally updates the tracker internal state.

    Args:
        model: Detection model instance
        tracker: Tracker instance
        json_file_path: Path to the JSON file containing detections
        video_source: Path to the video file
        update_tracker_state: Whether to update the tracker's internal state with loaded detections

    Returns:
        List of sv.Detections objects with loaded detections
    """
    if not os.path.exists(json_file_path):
        logger.error(f"JSON file not found: {json_file_path}")
        return []

    logger.info("Loading detections from JSON file.")

    rows_by_frame_number = defaultdict(list)
    with open(json_file_path, "r") as f:
        data = json.load(f)
    for row in data:
        frame_number = int(row["frame_id"])
        rows_by_frame_number[frame_number].append(row)

    detections_list = []

    video_info = VideoInfo.from_video_path(video_path=video_source)
    total_frames = video_info.total_frames or 0

    for frame_id in range(total_frames):
        if frame_id not in rows_by_frame_number.keys():
            detections = model.empty_detections()
        else:
            rows = rows_by_frame_number[frame_id]
            xyxy = []
            class_id = []
            confidence = []
            tracker_id = []
            custom_data = defaultdict(list)

            for row in rows:
                xyxy.append([row[key] for key in ["x_min", "y_min", "x_max", "y_max"]])
                class_id.append(row["class_id"])
                confidence.append(row["confidence"])
                tracker_id.append(row["tracker_id"])

                for custom_key in row.keys():
                    if custom_key in [
                        "x_min",
                        "y_min",
                        "x_max",
                        "y_max",
                        "class_id",
                        "confidence",
                        "tracker_id",
                        "frame_id",
                    ]:
                        continue
                    custom_data[custom_key].append(row[custom_key])

            if all([val == "" for val in class_id]):
                class_id = None
            if all([val == "" for val in confidence]):
                confidence = None
            if all([val == "" for val in tracker_id]):
                tracker_id = None

            detections = Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                class_id=np.array(class_id, dtype=int),
                confidence=np.array(confidence, dtype=np.float32),
                tracker_id=np.array(tracker_id, dtype=int),
                data=dict(custom_data),
            )

        detections.metadata.update({"frame_id": frame_id})
        detections_list.append(detections)

    # Update tracker state if requested
    if update_tracker_state and video_source:
        logger.info("Updating tracker state with loaded detections...")
        frames_generator = get_video_frames_generator(source_path=video_source)
        for frame_idx, (frame, detections) in enumerate(zip(frames_generator, detections_list)):
            tracker.update_tracks_with_loaded_detections(detections, frame)
            if frame_idx >= len(detections_list) - 1:
                break
        logger.info("Tracker state updated with loaded detections.")

    return detections_list


def load_serialized_detections(serialized_json: Dict[str, Any]) -> Detections:
    """Load detections from serialized format where all_detections contains JSON strings.

    This handles the format where tracking service saves individual serialized detection objects
    instead of the merged summary format.

    Args:
        serialized_json: Parsed JSON dict with 'all_detections' containing serialized strings

    Returns:
        Detections: Single merged detection object representing all rows.
    """
    try:
        import json
        all_detections = serialized_json.get('all_detections', [])
        if not all_detections:
            return Detections.empty()

        # Parse each serialized detection
        parsed_detections = []
        for serialized_str in all_detections:
            if isinstance(serialized_str, str):
                # Check if it's the old format (length-only strings)
                if serialized_str.startswith('<supervision_detections_object_len_'):
                    # Old format - can't deserialize, skip
                    continue
                else:
                    # Try to parse as JSON
                    try:
                        det_dict = json.loads(serialized_str)
                        parsed_detections.append(det_dict)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse serialized detection: {e}")
                        continue
            elif isinstance(serialized_str, dict):
                # Already a dict
                parsed_detections.append(serialized_str)
            else:
                continue

        if not parsed_detections:
            return Detections.empty()

        # Extract arrays from all detections
        xyxy_list = []
        confidence_list = []
        class_id_list = []
        tracker_id_list = []
        frame_index_list = []

        for det_dict in parsed_detections:
            xyxy = det_dict.get('xyxy', [])
            confidence = det_dict.get('confidence', [])
            class_id = det_dict.get('class_id', [])
            tracker_id = det_dict.get('tracker_id', [])
            data = det_dict.get('data', {})

            # Handle frame_index from data
            frame_index = data.get('frame_index', [])
            if isinstance(frame_index, list) and len(frame_index) > 0:
                frame_index_val = frame_index[0]  # Assume all detections in frame have same index
            else:
                frame_index_val = -1

            # Extend lists
            xyxy_list.extend(xyxy)
            confidence_list.extend(confidence)
            class_id_list.extend(class_id)
            if tracker_id:
                tracker_id_list.extend(tracker_id)
            frame_index_list.extend([frame_index_val] * len(xyxy))

        # Convert to numpy arrays
        xyxy = np.array(xyxy_list, dtype=np.float32) if xyxy_list else np.empty((0, 4), dtype=np.float32)
        class_id = np.array(class_id_list, dtype=int) if class_id_list else None
        confidence = np.array(confidence_list, dtype=np.float32) if confidence_list else None
        tracker_id = np.array(tracker_id_list, dtype=int) if tracker_id_list else None
        frame_index_array = np.array(frame_index_list, dtype=int) if frame_index_list else np.array([], dtype=int)

        # Create data dict with frame_index as numpy array
        data_field = {'frame_index': frame_index_array}

        # Cast to expected type to satisfy type checker
        from typing import cast, Dict, Union, List
        data_field = cast(Dict[str, Union[np.ndarray, List]], data_field)

        det = Detections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            tracker_id=tracker_id,
            data=data_field
        )
        return det
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to load serialized detections: {e}")
        return Detections.empty()
