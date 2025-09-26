"""
ID Generator utility functions for LaxAI project.

This module provides functions to generate unique IDs for various entities
used in the GCS structure, following the pattern: [type]_[GUID]
"""

import os
import uuid
from typing import Optional


def create_tenant_id() -> str:
    """Generate a unique tenant ID."""
    return f"tenant_{uuid.uuid4().hex[:8]}"


def create_video_id(video_path: Optional[str] = None) -> str:
    """Generate a unique video ID.
    
    Args:
        video_path: Optional path to video file. If provided, extracts and returns
                   just the filename. If not provided, generates UUID-based ID.
    """
    if video_path:
        # Extract filename from path (handles both Unix and Windows paths)
        # Split on both forward and backward slashes and take the last non-empty part
        parts = video_path.replace('\\', '/').split('/')
        filename = [part for part in parts if part][-1] if parts else video_path
        return filename
    else:
        return f"video_{uuid.uuid4().hex[:8]}"


def create_frame_id() -> str:
    """Generate a unique frame ID."""
    return f"frame_{uuid.uuid4().hex[:8]}"


def create_dataset_id() -> str:
    """Generate a unique dataset ID."""
    return f"dataset_{uuid.uuid4().hex[:8]}"


def create_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{uuid.uuid4().hex[:8]}"


def create_user_id() -> str:
    """Generate a unique user ID."""
    return f"user_{uuid.uuid4().hex[:8]}"


def create_experiment_id() -> str:
    """Generate a unique experiment ID."""
    return f"experiment_{uuid.uuid4().hex[:8]}"


def create_model_id() -> str:
    """Generate a unique model ID."""
    return f"model_{uuid.uuid4().hex[:8]}"


def create_crop_id(frame_id: str) -> str:
    """Generate a unique crop ID."""
    return f"crop_{frame_id}_{uuid.uuid4().hex[:8]}.jpg"


def create_aug_crop_id(crop_id: str, aug_idx: int) -> str:
    """Generate a unique augmented crop ID based on the original crop ID."""
    crop_id = crop_id.rstrip(".jpg")  # Remove file extension if present
    return f"{crop_id}_aug_{aug_idx}_{uuid.uuid4().hex[:8]}.jpg"


def create_batch_id() -> str:
    """Generate a unique batch ID."""
    return f"batch_{uuid.uuid4().hex[:8]}"


# Convenience function to create full GUIDs if needed
def create_full_guid() -> str:
    """Generate a full UUID without prefix."""
    return uuid.uuid4().hex


def create_short_guid() -> str:
    """Generate a shortened UUID (8 characters)."""
    return uuid.uuid4().hex[:8]
