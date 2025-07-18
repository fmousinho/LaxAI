"""
Functional training pipeline implementation.

This module provides a functional alternative to the class-based TrainPipeline
for simpler use cases where you don't need the full pipeline state management.
"""

import json
import logging
from typing import Dict, Any

from core.common.pipeline import Pipeline
from core.common.pipeline_step import PipelineStep
from core.train.train_pipeline import TrainPipeline
from config.all_config import DetectionConfig

logger = logging.getLogger(__name__)


def create_training_pipeline(tenant_id: str = "tenant1", 
                           delete_original_raw_videos: bool = False,
                           frames_per_video: int = 20,
                           verbose: bool = False,
                           save_intermediate: bool = False) -> TrainPipeline:
    """
    Factory function to create a configured training pipeline.
    
    Args:
        tenant_id: The tenant ID to process videos for
        delete_original_raw_videos: Whether to delete original raw video files after processing
        frames_per_video: Number of frames to extract per video
        verbose: Whether to enable verbose logging
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Configured TrainPipeline instance
    """
    config = DetectionConfig()
    config.delete_original_raw_videos = delete_original_raw_videos
    config.frames_per_video = frames_per_video
    
    return TrainPipeline(
        config=config,
        tenant_id=tenant_id,
        verbose=verbose,
        save_intermediate=save_intermediate
    )


def run_training_pipeline_simple(tenant_id: str = "tenant1", 
                                delete_original_raw_videos: bool = False,
                                frames_per_video: int = 20) -> Dict[str, Any]:
    """
    Simple functional interface to run the training pipeline with minimal configuration.
    
    Args:
        tenant_id: The tenant ID to process videos for
        delete_original_raw_videos: Whether to delete original raw video files after processing
        frames_per_video: Number of frames to extract per video
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline = create_training_pipeline(
        tenant_id=tenant_id,
        delete_original_raw_videos=delete_original_raw_videos,
        frames_per_video=frames_per_video,
        verbose=False,
        save_intermediate=False
    )
    
    return pipeline.run()


def run_training_pipeline_verbose(tenant_id: str = "tenant1", 
                                delete_original_raw_videos: bool = False,
                                frames_per_video: int = 20,
                                save_intermediate: bool = True) -> Dict[str, Any]:
    """
    Verbose functional interface to run the training pipeline with full logging and debugging.
    
    Args:
        tenant_id: The tenant ID to process videos for
        delete_original_raw_videos: Whether to delete original raw video files after processing
        frames_per_video: Number of frames to extract per video
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline = create_training_pipeline(
        tenant_id=tenant_id,
        delete_original_raw_videos=delete_original_raw_videos,
        frames_per_video=frames_per_video,
        verbose=True,
        save_intermediate=save_intermediate
    )
    
    return pipeline.run()


def get_training_pipeline_steps() -> Dict[str, str]:
    """
    Get information about the training pipeline steps.
    
    Returns:
        Dictionary mapping step names to descriptions
    """
    return {
        "import_video": "Import video from raw storage",
        "load_video": "Load video for processing",
        "extract_frames": "Extract frames with detections",
        "generate_detections": "Generate detections on frames",
        "extract_crops": "Extract and save crops from detections",
        "apply_masks": "Apply background masks to crops",
        "augment_crops": "Augment crops for training"
    }


def validate_training_pipeline_config(tenant_id: str, frames_per_video: int) -> bool:
    """
    Validate training pipeline configuration.
    
    Args:
        tenant_id: The tenant ID to validate
        frames_per_video: Number of frames per video to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    if not tenant_id or not isinstance(tenant_id, str):
        logger.error("Invalid tenant_id: must be a non-empty string")
        return False
    
    if not isinstance(frames_per_video, int) or frames_per_video <= 0:
        logger.error("Invalid frames_per_video: must be a positive integer")
        return False
    
    if frames_per_video > 100:
        logger.warning("frames_per_video is very high (>100), this may impact performance")
    
    return True


# For backward compatibility
run_training_pipeline = run_training_pipeline_simple
