"""
Detection utilities for the LaxAI project.

This module provides functions for converting detection objects to/from JSON format.
Supports both basic Detections serialization and enhanced serialization with RenderingConfig.
"""

import json
import logging
from typing import Dict, Any, List, Union, Optional, TYPE_CHECKING

import numpy as np
from supervision import Detections

if TYPE_CHECKING:
    from shared_libs.common.rendering_config import RenderingConfig

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


def detections_to_json(
    detections: Union[Detections, List[Any]],
    rendering_config: Optional['RenderingConfig'] = None,
    include_rendering_config: bool = False
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert detections into a list of JSON-serializable dictionaries.

    This implementation avoids repeatedly slicing the Supervision ``Detections`` object, which
    proved extremely slow (minutes) for large detection sets. Instead we iterate once over the
    underlying numpy arrays and build dictionaries directly.

    Args:
        detections: Either a ``Detections`` object or an empty list of detections.
        rendering_config: Optional RenderingConfig for styling/colors (only used if include_rendering_config=True)
        include_rendering_config: If True, returns dict with "detections" and "rendering_config" keys.
                                 If False (default), returns list of detection dicts (backward compatible)

    Returns:
        List of dictionaries ready for JSON serialization (default behavior)
        OR dict with "detections" and "rendering_config" keys (if include_rendering_config=True)
    """

    if not detections:
        if include_rendering_config:
            from shared_libs.common.rendering_config import RenderingConfig
            config = rendering_config or RenderingConfig()
            return {
                "detections": {
                    "xyxy": [],
                    "confidence": [],
                    "class_id": [],
                    "tracker_id": [],
                    "data": {},
                    "metadata": {}
                },
                "rendering_config": config.to_dict()
            }
        return []

    if isinstance(detections, list):
        if len(detections) == 0:
            if include_rendering_config:
                from shared_libs.common.rendering_config import RenderingConfig
                config = rendering_config or RenderingConfig()
                return {
                    "detections": {
                        "xyxy": [],
                        "confidence": [],
                        "class_id": [],
                        "tracker_id": [],
                        "data": {},
                        "metadata": {}
                    },
                    "rendering_config": config.to_dict()
                }
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

    # If rendering config requested, aggregate into single dict with config
    if include_rendering_config:
        from shared_libs.common.rendering_config import RenderingConfig
        
        # Combine all detections into single structure
        detections_dict: Dict[str, Any] = {
            "xyxy": [],
            "confidence": [],
            "class_id": [],
            "tracker_id": [],
            "data": {},
            "metadata": {}
        }
        
        for det in result:
            detections_dict["xyxy"].extend(det.get("xyxy", []))
            detections_dict["confidence"].extend(det.get("confidence", []))
            detections_dict["class_id"].extend(det.get("class_id", []))
            detections_dict["tracker_id"].extend(det.get("tracker_id", []))
            
            # Merge data
            for key, value in det.get("data", {}).items():
                if key not in detections_dict["data"]:
                    detections_dict["data"][key] = []
                if isinstance(value, list):
                    detections_dict["data"][key].extend(value)
                else:
                    detections_dict["data"][key].append(value)
            
            # Update metadata (take last)
            detections_dict["metadata"].update(det.get("metadata", {}))
        
        # Add rendering config
        config = rendering_config or RenderingConfig()
        return {
            "detections": detections_dict,
            "rendering_config": config.to_dict()
        }

    return result

def json_to_detections(
    json_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    return_rendering_config: bool = False
) -> Union[Detections, tuple[Detections, 'RenderingConfig']]:
    """Convert a list of JSON dictionaries back to a Detections object.
    
    Supports two input formats:
    1. List of detection dicts (backward compatible, original format)
    2. Dict with "detections" and "rendering_config" keys (new enhanced format)
    
    Args:
        json_data: Either:
                  - List of dictionaries containing detection data per frame (original)
                  - Dict with "detections" and "rendering_config" keys (enhanced)
        return_rendering_config: If True, returns tuple (Detections, RenderingConfig)
                                If False (default), returns only Detections (backward compatible)
        
    Returns:
        A single Detections object (default)
        OR tuple of (Detections, RenderingConfig) if return_rendering_config=True
    """
    from shared_libs.common.rendering_config import RenderingConfig
    
    rendering_config: Optional[RenderingConfig] = None
    json_list: List[Dict[str, Any]]
    
    # Handle new enhanced format with rendering_config
    if isinstance(json_data, dict) and "detections" in json_data:
        detections_data = json_data["detections"]
        
        # Convert single detections dict to list format
        json_list = [detections_data] if detections_data else []
        
        # Parse rendering config if present
        if "rendering_config" in json_data:
            rendering_config = RenderingConfig.from_dict(json_data["rendering_config"])
        
        # Process detections using existing logic below
        json_data_list = json_list
    else:
        # Original list format processing
        if not isinstance(json_data, list):
            json_data_list = [json_data]
        else:
            json_data_list = json_data
    
    if not json_data_list:
        detections = Detections.empty()
        if return_rendering_config:
            return detections, rendering_config or RenderingConfig()
        return detections
    
    # Collect all data across frames
    all_xyxy: List[Any] = []
    all_confidence: List[float] = []
    all_class_id: List[int] = []
    all_tracker_id: List[int] = []
    all_data: Dict[str, List[Any]] = {}
    all_metadata: Dict[str, Any] = {}
    
    for frame_data in json_data_list:
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
    
    # Return with or without rendering config
    if return_rendering_config:
        config_to_return = rendering_config if rendering_config is not None else RenderingConfig()
        return detection, config_to_return
    
    return detection


# ============================================================================
# Helper/convenience functions
# ============================================================================

def create_frame_response(
    frame_id: int,
    video_id: str,
    session_id: str,
    detections: Detections,
    rendering_config: Optional['RenderingConfig'] = None,
    has_next: bool = False,
    has_previous: bool = False,
    total_frames: int = 0
) -> Dict[str, Any]:
    """Create a complete frame annotation response with metadata.
    
    This is a convenience function for API responses that includes both
    detection/rendering data and navigation metadata.
    
    Args:
        frame_id: Frame identifier
        video_id: Video identifier
        session_id: Session identifier
        detections: supervision.Detections object
        rendering_config: Optional rendering configuration
        has_next: Whether there are more frames
        has_previous: Whether there are previous frames
        total_frames: Total number of frames
        
    Returns:
        Complete dictionary ready for API response
        
    Example:
        >>> response = create_frame_response(
        ...     frame_id=0,
        ...     video_id="test_video",
        ...     session_id="12345",
        ...     detections=detections,
        ...     rendering_config=config
        ... )
    """
    # Get detections and rendering config as dict
    data = detections_to_json(detections, rendering_config, include_rendering_config=True)
    
    # data should be a dict with "detections" and "rendering_config" when include_rendering_config=True
    if not isinstance(data, dict):
        raise TypeError("Expected dict from detections_to_json with include_rendering_config=True")
    
    # Add metadata
    result: Dict[str, Any] = {
        "frame_id": frame_id,
        "video_id": video_id,
        "session_id": session_id,
        "has_next": has_next,
        "has_previous": has_previous,
        "total_frames": total_frames
    }
    result.update(data)  # Add "detections" and "rendering_config"
    
    return result


def update_player_mapping(
    detections: Detections,
    rendering_config: 'RenderingConfig',
    tracker_id: int,
    new_player_id: int,
    style_preset: str = "default"
) -> tuple[Detections, 'RenderingConfig']:
    """Update player mapping for a tracker and apply style.
    
    This is a convenience function for updating both the detection data
    (player_id in data dict) and rendering config (style mapping) together.
    
    Args:
        detections: supervision.Detections object
        rendering_config: RenderingConfig object
        tracker_id: Tracker ID to update
        new_player_id: New player ID to assign
        style_preset: Style preset to apply (default, highlighted, etc.)
        
    Returns:
        Tuple of (updated_detections, updated_rendering_config)
        
    Example:
        >>> detections, config = update_player_mapping(
        ...     detections, config,
        ...     tracker_id=42,
        ...     new_player_id=7,
        ...     style_preset="highlighted"
        ... )
    """
    # Update detections data
    if detections.tracker_id is not None and "player_id" in detections.data:
        player_ids = detections.data["player_id"]
        for i, tid in enumerate(detections.tracker_id):
            if tid == tracker_id:
                player_ids[i] = new_player_id
    
    # Update rendering config
    rendering_config.set_style_for_player(new_player_id, style_preset)
    
    return detections, rendering_config
