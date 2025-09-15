"""
LaxAI Configuration Package

This package centralizes all configuration classes, instances, and transform
functions for the LaxAI system, providing a single point of access for all modules.
"""

# Import configuration instances and classes from all_config.py
from . import logging_config
from .all_config import BackgroundMaskConfig  # Instances; Classes
from .all_config import (ClusteringConfig, DebugConfig, DetectionConfig,
                         ModelConfig, PlayerConfig, TrackerConfig,
                         TrackStitchingConfig, TrainingConfig, TransformConfig,
                         background_mask_config, clustering_config,
                         debug_config, detection_config, model_config,
                         player_config, track_stitching_config, tracker_config,
                         training_config, transform_config)

# Lazily import transform-related functions and instances from transforms.py.
# The transforms module imports heavy dependencies (OpenCV / torchvision).
# Importing it at package import time causes those libraries to be loaded in
# environments (Cloud Run / Cloud Functions) where they may not be available.
#
# We use module-level __getattr__ (PEP 562) to defer importing until the
# attribute is actually accessed.
_TRANSFORM_EXPORTS = {
    "get_transforms",
    "TRANSFORMS",
    "training_transforms",
    "inference_transforms",
    "validation_transforms",
    "opencv_safe_transforms",
    "opencv_safe_training_transforms",
    "tensor_to_pil",
    "ensure_rgb_format",
    "BackgroundRemovalTransform",
    "create_background_removal_transform",
    "create_transforms_with_background_removal",
    "set_global_background_detector",
    "get_global_background_detector",
    "initialize_background_removal",
    "is_background_removal_enabled",
    "refresh_transform_instances",
    "create_background_detector_from_images",
}


def __getattr__(name: str):
    """Lazily load transform attributes from .transforms on first access.

    This prevents importing heavy ML/image libraries when simple config values
    are requested (for example, by services that only need configuration).
    """
    if name in _TRANSFORM_EXPORTS:
        from importlib import import_module

        mod = import_module(f"{__name__}.transforms")
        val = getattr(mod, name)
        # Cache on the module for subsequent attribute access
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    base = globals().keys()
    return sorted(list(base) + list(_TRANSFORM_EXPORTS))


# Import logging configuration (inferred from usage in dataprep_pipeline.py)


# Define the public API for the config package
__all__ = [
    # --- Configuration Instances ---
    "model_config",
    "tracker_config",
    "training_config",
    "detection_config",
    "clustering_config",
    "player_config",
    "track_stitching_config",
    "transform_config",
    "background_mask_config",
    "debug_config",
    "logging_config",
    # --- Configuration Classes ---
    "ModelConfig",
    "TrackerConfig",
    "TrainingConfig",
    "DetectionConfig",
    "ClusteringConfig",
    "PlayerConfig",
    "TrackStitchingConfig",
    "TransformConfig",
    "BackgroundMaskConfig",
    "DebugConfig",
    # --- Transform Functions & Components ---
    "get_transforms",  # type: ignore
    "create_transforms_with_background_removal",  # type: ignore
    "BackgroundRemovalTransform",  # type: ignore
    "create_background_removal_transform",  # type: ignore
    "ensure_rgb_format",  # type: ignore
    "tensor_to_pil",  # type: ignore
    # --- Transform Instances ---
    "TRANSFORMS",  # type: ignore
    "training_transforms",  # type: ignore
    "inference_transforms",  # type: ignore
    "validation_transforms",  # type: ignore
    "opencv_safe_transforms",  # type: ignore
    "opencv_safe_training_transforms",  # type: ignore
    # --- Background Removal API ---
    "set_global_background_detector",  # type: ignore
    "get_global_background_detector",  # type: ignore
    "initialize_background_removal",  # type: ignore
    "is_background_removal_enabled",  # type: ignore
    "refresh_transform_instances",  # type: ignore
    "create_background_detector_from_images",  # type: ignore
]
