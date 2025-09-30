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


def detection_to_json(detection: Detections) -> Dict[str, Any]:
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
        result = {
            "xyxy": detection.xyxy.tolist() if detection.xyxy is not None else [],
            "confidence": detection.confidence.tolist() if detection.confidence is not None else [],
            "class_id": detection.class_id.tolist() if detection.class_id is not None else [],
            "tracker_id": detection.tracker_id.tolist() if detection.tracker_id is not None else [],
            "data": {},
            "metadata": getattr(detection, 'metadata', {})
        }
        
        # Handle data field - convert numpy arrays to lists
        if hasattr(detection, 'data') and detection.data:
            for key, value in detection.data.items():
                if isinstance(value, np.ndarray):
                    result["data"][key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    result["data"][key] = list(value)
                else:
                    result["data"][key] = value
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to convert detection to JSON: {e}")
        raise ValueError(f"Cannot serialize detection object: {e}")


def json_to_detection(json_data: Dict[str, Any]) -> Detections:
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


def detections_list_to_json(detections_list: List[Detections]) -> List[Dict[str, Any]]:
    """Convert a list of Detections objects to a list of JSON-serializable dictionaries.
    
    Args:
        detections_list: List of supervision Detections objects
        
    Returns:
        List of dictionaries that can be serialized to JSON
    """
    return [detection_to_json(detection) for detection in detections_list]


def json_to_detections_list(json_list: List[Dict[str, Any]]) -> List[Detections]:
    """Convert a list of JSON dictionaries back to a list of Detections objects.
    
    Args:
        json_list: List of dictionaries containing detection data
        
    Returns:
        List of supervision Detections objects
    """
    return [json_to_detection(json_data) for json_data in json_list]
