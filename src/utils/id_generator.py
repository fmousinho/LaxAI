"""
ID Generator utility functions for LaxAI project.

This module provides functions to generate unique IDs for various entities
used in the GCS structure, following the pattern: [type]_[GUID]
"""

import uuid
from typing import Optional


def create_tenant_id() -> str:
    """Generate a unique tenant ID."""
    return f"tenant_{uuid.uuid4().hex[:8]}"


def create_video_id() -> str:
    """Generate a unique video ID."""
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
    crop_id = crop_id.replace('.jpg', '')  # Remove file extension if present
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


# Example usage and testing
if __name__ == "__main__":
    print("Generated IDs:")
    print(f"Tenant ID: {create_tenant_id()}")
    print(f"Video ID: {create_video_id()}")
    print(f"Frame ID: {create_frame_id()}")
    print(f"Dataset ID: {create_dataset_id()}")
    print(f"Run ID: {create_run_id()}")
    print(f"User ID: {create_user_id()}")
    print(f"Experiment ID: {create_experiment_id()}")
    print(f"Model ID: {create_model_id()}")
    print(f"Crop ID: {create_crop_id()}")
    print(f"Batch ID: {create_batch_id()}")
    
    print("\nExample GCS path usage:")
    tenant = create_tenant_id()
    video = create_video_id()
    frame = create_frame_id()
    dataset = create_dataset_id()
    run = create_run_id()
    
    print(f"Raw data path: {tenant}/raw/")
    print(f"Imported video path: {tenant}/process/{video}/imported/")
    print(f"Extracted frames path: {tenant}/process/{video}/{frame}/")
    print(f"Train dataset path: {tenant}/datasets/{dataset}/train")
    print(f"Run data path: {tenant}/runs/{run}/")
