"""
LaxAI Configuration Package

This package centralizes all configuration classes, instances, and transform
functions for the LaxAI system, providing a single point of access for all modules.
"""

# Import configuration instances and classes from all_config.py
from .all_config import (
    # Instances
    model_config,
    tracker_config,
    training_config,
    detection_config,
    clustering_config,
    player_config,
    track_stitching_config,
    transform_config,
    background_mask_config,
    debug_config,
    # Classes
    ModelConfig,
    TrackerConfig,
    TrainingConfig,
    DetectionConfig,
    ClusteringConfig,
    PlayerConfig,
    TrackStitchingConfig,
    TransformConfig,
    BackgroundMaskConfig,
    DebugConfig
)

# Import transform-related functions and instances from transforms.py
from .transforms import (
    # Main transform getter
    get_transforms,
    # Transform instances
    TRANSFORMS,
    training_transforms,
    inference_transforms,
    validation_transforms,
    opencv_safe_transforms,
    opencv_safe_training_transforms,
    # Utility functions
    tensor_to_pil,
    ensure_rgb_format,
    # Background removal components
    BackgroundRemovalTransform,
    create_background_removal_transform,
    create_transforms_with_background_removal,
    # Global background detector management
    set_global_background_detector,
    get_global_background_detector,
    initialize_background_removal,
    is_background_removal_enabled,
    refresh_transform_instances,
    create_background_detector_from_images
)

# Import logging configuration (inferred from usage in dataprep_pipeline.py)
from . import logging_config


# Define the public API for the config package
__all__ = [
    # --- Configuration Instances ---
    'model_config',
    'tracker_config',
    'training_config',
    'detection_config',
    'clustering_config',
    'player_config',
    'track_stitching_config',
    'transform_config',
    'background_mask_config',
    'debug_config',
    'logging_config',

    # --- Configuration Classes ---
    'ModelConfig',
    'TrackerConfig',
    'TrainingConfig',
    'DetectionConfig',
    'ClusteringConfig',
    'PlayerConfig',
    'TrackStitchingConfig',
    'TransformConfig',
    'BackgroundMaskConfig',
    'DebugConfig',

    # --- Transform Functions & Components ---
    'get_transforms',
    'create_transforms_with_background_removal',
    'BackgroundRemovalTransform',
    'create_background_removal_transform',
    'ensure_rgb_format',
    'tensor_to_pil',

    # --- Transform Instances ---
    'TRANSFORMS',
    'training_transforms',
    'inference_transforms',
    'validation_transforms',
    'opencv_safe_transforms',
    'opencv_safe_training_transforms',

    # --- Background Removal API ---
    'set_global_background_detector',
    'get_global_background_detector',
    'initialize_background_removal',
    'is_background_removal_enabled',
    'refresh_transform_instances',
    'create_background_detector_from_images',
]