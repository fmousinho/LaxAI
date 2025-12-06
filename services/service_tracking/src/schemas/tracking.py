"""
Tracking API request/response schemas for the tracking service.
Explicit Pydantic models that use config defaults directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from shared_libs.config.all_config import (
    detection_config,
    tracker_config,
)


class TrackingParams(BaseModel):
    """Tracking hyperparameters - mirrors DetectionConfig and TrackerConfig."""

    # Detection Config
    nms_iou_threshold: Optional[float] = Field(default=detection_config.nms_iou_threshold, description="NMS IOU threshold")
    prediction_threshold: float = Field(default=detection_config.prediction_threshold, description="Prediction confidence threshold")
    model_checkpoint: str = Field(default=detection_config.model_checkpoint, description="Detection model checkpoint name")
    
    # Tracker Config
    track_activation_threshold: float = Field(default=tracker_config.track_activation_threshold, description="Track activation threshold")
    lost_track_buffer: int = Field(default=tracker_config.lost_track_buffer, description="Lost track buffer size")
    minimum_matching_threshold: float = Field(default=tracker_config.minimum_matching_threshold, description="Minimum matching threshold")
    minimum_consecutive_frames: int = Field(default=tracker_config.minimum_consecutive_frames, description="Minimum consecutive frames for track")
    
    # Pipeline Config
    video_limit: Optional[int] = Field(default=None, description="Maximum number of videos to process")
    resume_from_checkpoint: bool = Field(default=True, description="Resume from checkpoint if available")
    
    # Model Config
    model_input_width: int = Field(default=224, description="Model input width")
    model_input_height: int = Field(default=224, description="Model input height")


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
