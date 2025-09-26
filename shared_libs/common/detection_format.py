"""
Detection data structures and formats.

This module provides minimal data structures for detection results,
avoiding heavy dependencies like supervision.
"""

from typing import Dict, Optional


class Detections:
    """Minimal Detections class that mimics supervision.Detections interface."""

    def __init__(self, xyxy=None, mask=None, confidence=None, class_id=None,
                 tracker_id=None, data=None, metadata=None):
        self.xyxy = xyxy
        self.mask = mask
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.data = data or {}
        self.metadata = metadata or {}

    @classmethod
    def empty(cls):
        return cls()