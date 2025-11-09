"""
LaxAI Configuration Package

This package centralizes all configuration classes, instances, and transform
functions for the LaxAI system, providing a single point of access for all modules.
"""

# Import configuration instances and classes from all_config.py
from .all_config import BackgroundMaskConfig  # Instances; Classes
from .all_config import (ClusteringConfig, DebugConfig, DetectionConfig,
                         ModelConfig, PlayerConfig, TrackerConfig,
                         TrackStitchingConfig, TrainingConfig, TransformConfig,
                         background_mask_config, clustering_config,
                         debug_config, detection_config, model_config,
                         player_config, track_stitching_config, tracker_config,
                         training_config, transform_config)

# Import transform-related functions and instances from shared_libs
from shared_libs.config.transforms import (
    BackgroundRemovalTransform,
    TRANSFORMS,
    create_background_detector_from_images,
    create_background_removal_transform,
    create_transforms_with_background_removal,
    ensure_rgb_format,
    get_global_background_detector,
    get_transforms,
    inference_transforms,
    initialize_background_removal,
    is_background_removal_enabled,
    opencv_safe_training_transforms,
    opencv_safe_transforms,
    refresh_transform_instances,
    set_global_background_detector,
    tensor_to_pil,
    training_transforms,
    validation_transforms,
)

# Import logging configuration (inferred from usage in dataprep_pipeline.py)
from shared_libs.config import logging_config

# Define the public API for the config package
__all__ = [  # type: ignore[reportUnsupportedDunderAll]
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

    # --- Transform Functions and Classes (lazy-loaded) ---
    'get_transforms',  # type: ignore[reportUnsupportedDunderAll]
    'create_transforms_with_background_removal',  # type: ignore[reportUnsupportedDunderAll]
    'BackgroundRemovalTransform',  # type: ignore[reportUnsupportedDunderAll]
    'create_background_removal_transform',  # type: ignore[reportUnsupportedDunderAll]
    'ensure_rgb_format',  # type: ignore[reportUnsupportedDunderAll]
    'tensor_to_pil',  # type: ignore[reportUnsupportedDunderAll]

    # --- Transform Constants ---
    'TRANSFORMS',  # type: ignore[reportUnsupportedDunderAll]
    'training_transforms',  # type: ignore[reportUnsupportedDunderAll]
    'inference_transforms',  # type: ignore[reportUnsupportedDunderAll]
    'validation_transforms',  # type: ignore[reportUnsupportedDunderAll]
    'opencv_safe_transforms',  # type: ignore[reportUnsupportedDunderAll]
    'opencv_safe_training_transforms',  # type: ignore[reportUnsupportedDunderAll]

    # --- Background Removal Functions ---
    'set_global_background_detector',  # type: ignore[reportUnsupportedDunderAll]
    'get_global_background_detector',  # type: ignore[reportUnsupportedDunderAll]
    'initialize_background_removal',  # type: ignore[reportUnsupportedDunderAll]
    'is_background_removal_enabled',  # type: ignore[reportUnsupportedDunderAll]
    'refresh_transform_instances',  # type: ignore[reportUnsupportedDunderAll]
    'create_background_detector_from_images',  # type: ignore[reportUnsupportedDunderAll]
]