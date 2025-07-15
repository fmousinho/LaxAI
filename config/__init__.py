"""
LaxAI Configuration Package

This package contains configuration classes and transform functions for the LaxAI system.
"""

from .all_config import (
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

from .transforms import (
    get_transforms,
    create_transforms_with_background_removal,
    BackgroundRemovalTransform,
    create_background_removal_transform,
    ensure_rgb_format,
    TRANSFORMS,
    training_transforms,
    inference_transforms,
    validation_transforms,
    tensor_to_pil,
    opencv_safe_transforms,
    opencv_safe_training_transforms,
    # New centralized background removal functions
    set_global_background_detector,
    get_global_background_detector,
    initialize_background_removal,
    is_background_removal_enabled,
    refresh_transform_instances,
    create_background_detector_from_images
)

__all__ = [
    # Configuration instances
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
    # Configuration classes
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
    # Transform functions
    'get_transforms',
    'create_transforms_with_background_removal',
    'BackgroundRemovalTransform',
    'create_background_removal_transform',
    'ensure_rgb_format',
    'set_global_background_detector',
    'get_global_background_detector',
    'initialize_background_removal',
    'is_background_removal_enabled',
    'refresh_transform_instances',
    'create_background_detector_from_images',
    # Transform instances
    'TRANSFORMS',
    'training_transforms',
    'inference_transforms',
    'validation_transforms',
    'tensor_to_pil',
    'opencv_safe_transforms',
    'opencv_safe_training_transforms'
]