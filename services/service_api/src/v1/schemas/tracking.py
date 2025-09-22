"""
Tracking API schemas for LaxAI API Service.

Complete schemas for API request/response validation with specific parameter fields.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrackingParams(BaseModel):
    """Tracking-specific parameters."""

    verbose: Optional[bool] = Field(True, description="Enable verbose logging for pipelines")
    save_intermediate: Optional[bool] = Field(True, description="Save intermediate pipeline results to GCS")
    resume_from_checkpoint: Optional[bool] = Field(True, description="Resume track generation from checkpoint if available")
    video_limit: Optional[int] = Field(None, description="Maximum number of videos to process (None for all)")


class TrackingRequest(BaseModel):
    """Tracking request model for API services."""

    custom_name: str = Field(
        default="track_generation_workflow_run",
        description="Custom name for the tracking run"
    )
    tenant_id: str = Field(
        ...,
        description="Tenant identifier for the tracking job"
    )
    tracking_params: Optional[TrackingParams] = Field(
        default=None,
        description="Tracking-specific parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "custom_name": "my_tracking_run",
                "tenant_id": "default",
                "tracking_params": {
                    "verbose": True,
                    "save_intermediate": True,
                    "resume_from_checkpoint": True,
                    "video_limit": None
                }
            }
        }


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
    videos_found: Optional[int] = Field(None, description="Number of videos discovered")
    videos_processed: Optional[int] = Field(None, description="Number of videos processed")
    successful_runs: Optional[int] = Field(None, description="Number of successful track generation runs")
    total_runs: Optional[int] = Field(None, description="Total number of track generation runs")
    current_video: Optional[str] = Field(None, description="Currently processing video path")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Tracking metrics")
    logs: Optional[List[str]] = Field(None, description="Recent tracking logs")
    updated_at: str = Field(..., description="Last update timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")