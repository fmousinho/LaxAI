"""Lightweight serialization utilities for Detections objects.

This module replaces the previous dependency on `supervision.JSONSink` by
providing minimal helpers to serialize our custom `Detections` class (defined
in `detection_format.py`) or any supervision-like object exposing similar
attributes.

Functions
---------
serialize_detections(obj) -> str
    Convert a Detections-like object into a JSON string, omitting None fields
    and converting numpy arrays to lists.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict
import json
import logging

logger = logging.getLogger(__name__)

__all__ = ["serialize_detections", "detections_to_dict"]


def _to_serializable(value: Any):
    """Recursively convert value into JSON-serializable primitives."""
    import numpy as _np  # local import to avoid imposing at module load

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, _np.ndarray):  # type: ignore
        return value.tolist()  # type: ignore
    return str(value)


def detections_to_dict(det_obj: Any) -> Dict[str, Any]:
    """Convert a Detections-like object to a dictionary.

    The function is attribute-driven and does not import supervision.
    """
    if is_dataclass(det_obj):  # type: ignore
        raw = asdict(det_obj)  # type: ignore
    else:
        attr_names = [
            "xyxy",
            "mask",
            "confidence",
            "class_id",
            "tracker_id",
            "data",
            "metadata",
        ]
        raw = {}
        for name in attr_names:
            if hasattr(det_obj, name):
                raw[name] = getattr(det_obj, name)
    return {k: _to_serializable(v) for k, v in raw.items() if v is not None}


def serialize_detections(det_obj: Any) -> str:
    """Serialize a Detections-like object into JSON.

    Returns an empty JSON object string '{}' on error.
    """
    try:
        payload = detections_to_dict(det_obj)
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Failed to serialize detections object: {e}")
        return "{}"
