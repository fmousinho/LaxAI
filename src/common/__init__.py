"""
LaxAI Common Utilities Package

This package provides shared components used across the application, including:
- Google Cloud Storage client
- Base pipeline and step classes
- Detection, tracking, and cropping utilities
"""

from .background_mask import (
    BackgroundMaskDetector,
    create_frame_generator_from_images
)
from .crop_utils import (
    extract_crops_from_video,
    create_train_val_split,
    reorganize_crops_by_stitched_tracks
)
from .detection import DetectionModel
from .detection_utils import (
    process_frames,
    load_detections_from_json
)
from .google_storage import (
    get_storage,
    GoogleStorageClient,
    GCSPaths
)
from .pipeline import (
    Pipeline,
    PipelineStatus
)
from .pipeline_step import (
    PipelineStep,
    StepStatus
)

__all__ = [
    # background_mask
    'BackgroundMaskDetector',
    'create_frame_generator_from_images',
    # crop_utils
    'extract_crops_from_video',
    'create_train_val_split',
    'reorganize_crops_by_stitched_tracks',
    # detection
    'DetectionModel',
    # detection_utils
    'process_frames',
    'load_detections_from_json',
    # google_storage
    'get_storage',
    'GoogleStorageClient',
    'GoogleStorageConfig',
    # pipeline
    'Pipeline',
    'PipelineStatus',
    # pipeline_step
    'PipelineStep',
    'StepStatus',
]