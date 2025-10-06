"""
Detection utilities for the LaxAI project.

This module provides functions for converting detection objects to/from JSON format.
"""

import json
import logging
from typing import Dict, Any, List, Union, Optional

import numpy as np
from supervision import Detections

logger = logging.getLogger(__name__)


def detection_to_json_single(detection: Detections) -> Dict[str, Any]:
    """Convert a Detections object to a JSON-serializable dictionary.
    
    Args:
        detection: A supervision Detections object
        
    Returns:
        Dictionary that can be serialized to JSON
        
    Raises:
        ValueError: If detection object is invalid or cannot be serialized
    """
    try:
        # Handle empty detections
        if len(detection) == 0:
            return {
                "xyxy": [],
                "confidence": [],
                "class_id": [],
                "tracker_id": [],
                "data": {},
                "metadata": getattr(detection, 'metadata', {})
            }
        
        # Convert numpy arrays to lists for JSON serialization
        # For single detections, iterate to get the actual values
        xyxy_val, mask, conf_val, class_val, tracker_val, data_dict = next(iter(detection))
        
        result = {
            "xyxy": [xyxy_val.tolist()],
            "confidence": [float(conf_val)] if conf_val is not None else [],
            "class_id": [int(class_val)] if class_val is not None else [],
            "tracker_id": [int(tracker_val)] if tracker_val is not None else [],
            "data": {},
            "metadata": getattr(detection, 'metadata', {})
        }
        
        # Handle data field from iteration
        if data_dict:
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    converted = value.tolist()
                    # Ensure we store a flat list rather than a nested structure
                    result["data"][key] = converted if isinstance(converted, list) else [converted]
                elif isinstance(value, (list, tuple)):
                    result["data"][key] = list(value)
                else:
                    result["data"][key] = [value]
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to convert detection to JSON: {e}")
        raise ValueError(f"Cannot serialize detection object: {e}")


def json_to_detection_single(json_data: Dict[str, Any]) -> Detections:
    """Convert a JSON dictionary back to a Detections object.
    
    Args:
        json_data: Dictionary containing detection data (from detection_to_json)
        
    Returns:
        A supervision Detections object
        
    Raises:
        ValueError: If JSON data is invalid or cannot be converted
    """
    try:
        # Extract arrays and convert to numpy arrays
        xyxy = np.array(json_data.get("xyxy", []), dtype=np.float32)
        confidence = json_data.get("confidence")
        class_id = json_data.get("class_id")
        tracker_id = json_data.get("tracker_id")
        
        # Handle empty detections
        if len(xyxy) == 0:
            xyxy = np.empty((0, 4), dtype=np.float32)
        else:
            # Ensure xyxy has correct shape
            if xyxy.ndim == 1:
                xyxy = xyxy.reshape(-1, 4)
        
        # Convert other arrays to numpy with proper types and handle None values
        confidence_array = None
        if confidence is not None and len(confidence) > 0:
            confidence_array = np.array(confidence, dtype=np.float32)
        
        class_id_array = None
        if class_id is not None and len(class_id) > 0:
            class_id_array = np.array(class_id, dtype=int)
        
        tracker_id_array = None
        if tracker_id is not None and len(tracker_id) > 0:
            tracker_id_array = np.array(tracker_id, dtype=int)
        
        # Handle data field - convert lists back to numpy arrays where appropriate
        data_dict = {}
        if "data" in json_data and json_data["data"]:
            for key, value in json_data["data"].items():
                if isinstance(value, list) and len(value) > 0:
                    # Try to convert to numpy array if it's numeric data
                    try:
                        if all(isinstance(x, (int, float)) for x in value):
                            data_dict[key] = np.array(value)
                        else:
                            data_dict[key] = value
                    except (ValueError, TypeError):
                        data_dict[key] = value
                else:
                    data_dict[key] = value
        
        # Create the Detections object
        detection = Detections(
            xyxy=xyxy,
            confidence=confidence_array,
            class_id=class_id_array,
            tracker_id=tracker_id_array,
            data=data_dict if data_dict else {}
        )
        
        # Set metadata if present
        if "metadata" in json_data and json_data["metadata"]:
            detection.metadata.update(json_data["metadata"])
        
        return detection
        
    except Exception as e:
        logger.error(f"Failed to convert JSON to detection: {e}")
        raise ValueError(f"Cannot deserialize JSON data to detection object: {e}")


def _normalize_data_value(value: Any) -> List[Any]:
    """Normalize detection auxiliary data into a JSON-friendly list."""

    if value is None:
        return []

    if isinstance(value, np.ndarray):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]

    if isinstance(value, np.generic):  # numpy scalar
        return [value.item()]

    if isinstance(value, (list, tuple)):
        normalized: List[Any] = []
        for item in value:
            if isinstance(item, np.ndarray):
                item_converted = item.tolist()
                normalized.append(item_converted if isinstance(item_converted, list) else [item_converted])
            elif isinstance(item, np.generic):
                normalized.append(item.item())
            else:
                normalized.append(item)
        return normalized

    return [value]


def detections_to_json(detections: Union[Detections, List[Any]]) -> List[Dict[str, Any]]:
    """Convert detections into a list of JSON-serializable dictionaries.

    This implementation avoids repeatedly slicing the Supervision ``Detections`` object, which
    proved extremely slow (minutes) for large detection sets. Instead we iterate once over the
    underlying numpy arrays and build dictionaries directly.

    Args:
        detections: Either a ``Detections`` object or an empty list of detections.

    Returns:
        List of dictionaries ready for JSON serialization.
    """

    if not detections:
        return []

    if isinstance(detections, list):
        if len(detections) == 0:
            return []
        if len(detections) == 1 and isinstance(detections[0], Detections):
            detections = detections[0]
        else:
            raise TypeError("detections_to_json expects a Detections object or an empty list")

    if not isinstance(detections, Detections):
        raise TypeError("detections_to_json expects a Detections object")

    metadata = dict(getattr(detections, "metadata", {}) or {})

    result: List[Dict[str, Any]] = []
    # Iterate over detections once; ``supervision.Detections`` yields per-item tuples
    for xyxy_val, _mask, conf_val, class_val, tracker_val, data_dict in detections:  # type: ignore[misc]
        entry: Dict[str, Any] = {
            "xyxy": [xyxy_val.tolist()],
            "confidence": [float(conf_val)] if conf_val is not None else [],
            "class_id": [int(class_val)] if class_val is not None else [],
            "tracker_id": [int(tracker_val)] if tracker_val is not None else [],
            "data": {},
            "metadata": metadata,
        }

        if data_dict:
            for key, value in data_dict.items():
                entry["data"][key] = _normalize_data_value(value)

        result.append(entry)

    return result

def json_to_detections(json_list: List[Dict[str, Any]]) -> Detections:
    """Convert a list of JSON dictionaries back to a Detections object.
    
    Each dictionary in the list represents detections from one frame.
    This function concatenates all detections from all frames into a single Detections object.
    
    Args:
        json_list: List of dictionaries containing detection data per frame
        
    Returns:
        A single Detections object with all detections concatenated
    """
    if not json_list:
        return Detections.empty()
    
    # Collect all data across frames
    all_xyxy = []
    all_confidence = []
    all_class_id = []
    all_tracker_id = []
    all_data = {}
    all_metadata = {}
    
    for frame_data in json_list:
        # Extract frame-level data
        xyxy = frame_data.get("xyxy", [])
        confidence = frame_data.get("confidence", [])
        class_id = frame_data.get("class_id", [])
        tracker_id = frame_data.get("tracker_id", [])
        data = frame_data.get("data", {})
        metadata = frame_data.get("metadata", {})
        
        # Extend arrays
        all_xyxy.extend(xyxy)
        all_confidence.extend(confidence)
        all_class_id.extend(class_id)
        all_tracker_id.extend(tracker_id)
        
        # Merge data dictionaries (extend lists, etc.)
        for key, value in data.items():
            if key not in all_data:
                all_data[key] = []
            if isinstance(value, list):
                all_data[key].extend(value)
            else:
                all_data[key].append(value)
        
        # For metadata, we might want to track per-frame, but for now just take the last one
        all_metadata.update(metadata)
    
    # Convert to numpy arrays
    xyxy_array = np.array(all_xyxy, dtype=np.float32) if all_xyxy else np.empty((0, 4), dtype=np.float32)
    confidence_array = np.array(all_confidence, dtype=np.float32) if all_confidence else None
    class_id_array = np.array(all_class_id, dtype=int) if all_class_id else None
    tracker_id_array = np.array(all_tracker_id, dtype=int) if all_tracker_id else None
    
    # Convert data lists to numpy arrays where appropriate
    data_dict = {}
    for key, value in all_data.items():
        if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
            data_dict[key] = np.array(value)
        else:
            data_dict[key] = value
    
    # Create the Detections object
    detection = Detections(
        xyxy=xyxy_array,
        confidence=confidence_array,
        class_id=class_id_array,
        tracker_id=tracker_id_array,
        data=data_dict
    )
    
    # Set metadata
    detection.metadata.update(all_metadata)
    
    return detection
