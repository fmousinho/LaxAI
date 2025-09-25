"""
LaxAI Common Utilities Package

This package provides shared components used across the application, including:
- Google Cloud Storage client
- Base pipeline and step classes
- Detection, tracking, and cropping utilities
"""

from .background_mask import (BackgroundMaskDetector,
                              create_frame_generator_from_images)
from .crop_utils import (create_train_val_split, extract_crops_from_video,
                         reorganize_crops_by_stitched_tracks)
from .detection import DetectionModel
from .detection_utils import (load_detections_from_json, process_frames,
                              save_all_detections, load_all_detections_summary)
from .google_storage import GoogleStorageClient, get_storage
from .pipeline import Pipeline, PipelineStatus
from .pipeline_step import PipelineStep, StepStatus
from .player import Player
from .track_to_player import (get_player_ids_from_detections,
                              get_unique_players_from_detections,
                              map_detections_to_players)
from .tracker import AffineAwareByteTrack

__all__ = [
    # background_mask
    "BackgroundMaskDetector",
    "create_frame_generator_from_images",
    # crop_utils
    "extract_crops_from_video",
    "create_train_val_split",
    "reorganize_crops_by_stitched_tracks",
    # detection
    "DetectionModel",
    # detection_utils
    "process_frames",
    "load_detections_from_json",
    "save_all_detections",
    "load_all_detections_summary",
    # google_storage
    "get_storage",
    "GoogleStorageClient",
    # pipeline
    "Pipeline",
    "PipelineStatus",
    # pipeline_step
    "PipelineStep",
    "StepStatus",
    # player
    "Player",
    # track_to_player
    "map_detections_to_players",
    "get_player_ids_from_detections",
    "get_unique_players_from_detections",
    # tracker
    "AffineAwareByteTrack",
]
