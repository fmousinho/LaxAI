"""
Tracking API request/response schemas and configuration for the tracking service.
Explicit Pydantic models that use config defaults directly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared_libs.config.all_config import (
    tracker_config,
)


class DetectionModelConfig(BaseSettings):
    """Configuration for the detection model, including WandB artifact settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # WandB settings
    wandb_api_key: str = Field(
        ...,
        description="WandB API key for authentication",
    )
    wandb_model_artifact: str = Field(
        default="fmousinho76-home-org/wandb-registry-model/Detections:latest",
        description="WandB artifact path for the detection model",
    )
    wandb_project: str = Field(
        default="LaxAI-Tracking",
        description="WandB project name for tracking model downloads",
    )
    wandb_run_name: str = Field(
        default="model-download",
        description="WandB run name for model download operations",
    )

    # Model file settings
    artifact_file_name: str = Field(
        default="common-models-detection_latest.pth",
        description="Filename of the model weights within the artifact",
    )
    num_classes: int = Field(
        default=6,
        description="Number of classes the model was trained with",
    )

    # Device settings
    device: Optional[str] = Field(
        default=None,
        description="Device to run the model on (cuda/mps/cpu). Auto-detected if None.",
    )


# Global detection model config instance
detection_model_config = DetectionModelConfig()



class TrackingParams(BaseModel):
    """Tracking hyperparameters - mirrors DetectionConfi    g and TrackerConfig."""

    # Detection Config
    prediction_threshold: float = Field(default=tracker_config.prediction_threshold, description="Prediction confidence threshold")
    
    # Tracker Config
    track_activation_threshold: float = Field(default=tracker_config.track_activation_threshold, description="Used to separate high and low confidence detections")
    lost_track_buffer: int = Field(default=tracker_config.lost_track_buffer, description="How many frames to wait before declaring a track lost")
    max_match_distance: float = Field(default=tracker_config.max_match_distance, description="Max distance for tracks to be matched")
    min_consecutive_frames: int = Field(default=tracker_config.min_consecutive_frames, description="Minimum consecutive frames for tracks to be confirmed")
    
    # Embedding Config
    embedding_update_frequency: int = Field(default=5, description="Frequency (in frames) to update track embeddings")
    embedding_quality_threshold: float = Field(default=0.6, description="Minimum detection score to trigger embedding update")
    embedding_min_detection_confidence: float = Field(default=0.6, description="Minimum confidence for detections to be considered for embeddings")
    
    # Pipeline Config
    resume_from_checkpoint: bool = Field(default=True, description="Resume from checkpoint if available")
    



class TrackingRequest(BaseModel):
    """Tracking request model with explicit parameter validation."""

    tenant_id: str = Field(..., description="Tenant identifier for the tracking job")
    custom_name: str = Field(default="track_generation_run", description="Custom name for the run")
    tracking_params: Optional[TrackingParams] = Field(default_factory=TrackingParams, description="Tracking-specific parameters")
    verbose: bool = Field(default=True, description="Enable verbose logging")


class TrackingResponse(BaseModel):
    """Tracking response model."""

    task_id: str = Field(..., description="Unique identifier for the tracking task")
    status: str = Field(..., description="Current status of the tracking task")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="Task creation timestamp")


class TrackingStatus(BaseModel):
    """Tracking status model."""

    task_id: str = Field(..., description="Unique identifier for the tracking task")
    status: str = Field(..., description="Current status of the tracking task")
    progress: Optional[float] = Field(None, description="Tracking progress percentage (0-100)")
    frames_processed: Optional[int] = Field(None, description="Number of frames processed")
    total_frames: Optional[int] = Field(None, description="Total number of frames")
    videos_processed: Optional[int] = Field(None, description="Number of videos processed")
    total_videos: Optional[int] = Field(None, description="Total number of videos")
    error: Optional[str] = Field(None, description="Error message if failed")
    updated_at: str = Field(..., description="Last update timestamp")
