"""
DataPrep API schemas for LaxAI DataPrep Service.

Complete schemas for API request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessFoldersResponse(BaseModel):
    """Response model for listing process folders."""

    folders: List[str] = Field(..., description="List of available process folder names (video IDs)")


class StartPrepRequest(BaseModel):
    """Request model for starting a verification session."""

    video_id: str = Field(..., description="Video ID to start verification for (can be full path, filename with .mp4, or filename without extension)")

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "GRIT Dallas-Houston 2027 vs Capital 2027 Orange - 9-00am_summary"
            }
        }


class StartPrepResponse(BaseModel):
    """Response model for starting a verification session."""

    success: bool = Field(..., description="Whether the session was started successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class VerificationImagesResponse(BaseModel):
    """Response model for getting images for verification."""

    status: str = Field(..., description="Status of the verification process")
    message: Optional[str] = Field(None, description="Optional message")
    pair_id: Optional[str] = Field(None, description="Identifier for the verification pair (e.g., '3-120')")
    group1_id: Optional[int] = Field(None, description="First group ID for verification")
    group2_id: Optional[int] = Field(None, description="Second group ID for verification")
    mode: Optional[str] = Field(None, description="Verification mode for this pair (normal, second_pass, skipped_only)")
    issued_at: Optional[datetime] = Field(None, description="Timestamp when the pair was issued")
    expires_at: Optional[datetime] = Field(None, description="Timestamp when the pair will expire if not answered")
    group1_prefixes: Optional[List[str]] = Field(None, description="GCS prefixes for group1 tracks")
    group2_prefixes: Optional[List[str]] = Field(None, description="GCS prefixes for group2 tracks")
    total_pairs: Optional[int] = Field(None, description="Total number of possible pairs")
    verified_pairs: Optional[int] = Field(None, description="Number of pairs already verified")
    outstanding_pair_ids: Optional[List[str]] = Field(None, description="List of currently outstanding pair IDs")
    max_outstanding_pairs: Optional[int] = Field(None, description="Maximum number of outstanding pairs allowed")


class RecordResponseRequest(BaseModel):
    """Request model for recording a user response."""

    pair_id: str = Field(..., description="Identifier of the verification pair being responded to")
    decision: str = Field(..., description="User's decision: 'same', 'different', or 'skip'")

    class Config:
        json_schema_extra = {
            "example": {
                "pair_id": "3-120",
                "decision": "same"
            }
        }


class RecordResponseResponse(BaseModel):
    """Response model for recording a user response."""

    success: bool = Field(..., description="Whether the response was recorded successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")
    pair_id: Optional[str] = Field(None, description="Identifier of the verification pair that was processed")
    pair_status: Optional[str] = Field(None, description="Status of the verification pair after processing")
    outstanding_pair_ids: Optional[List[str]] = Field(None, description="List of currently outstanding pair IDs")
    max_outstanding_pairs: Optional[int] = Field(None, description="Maximum number of outstanding pairs allowed")


class SaveGraphResponse(BaseModel):
    """Response model for saving the graph."""

    success: bool = Field(..., description="Whether the graph was saved successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class SaveGraphImageResponse(BaseModel):
    """Response model for saving the graph image."""

    success: bool = Field(..., description="Whether the graph image was saved successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")
    image_url: Optional[str] = Field(None, description="GCS URL of the saved graph image (gs://bucket/path)")


class SuspendPrepResponse(BaseModel):
    """Response model for suspending a verification session."""

    success: bool = Field(..., description="Whether the session was suspended successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class MoveCropsResponse(BaseModel):
    """Response model for moving crops to verified tracks."""

    success: bool = Field(..., description="Whether all crops were moved successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class SplitTrackRequest(BaseModel):
    """Request model for splitting a track at a specific frame."""

    track_id: int = Field(..., description="The track ID to split")
    crop_image_name: str = Field(..., description="Name of the crop image where the split occurs (e.g., 'crop_960.jpg')")

    class Config:
        json_schema_extra = {
            "example": {
                "track_id": 1,
                "crop_image_name": "crop_960.jpg"
            }
        }


class SplitTrackResponse(BaseModel):
    """Response model for splitting a track."""

    success: bool = Field(..., description="Whether the track was split successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class GraphStatisticsResponse(BaseModel):
    """Response model for graph statistics."""

    success: bool = Field(..., description="Whether the statistics were retrieved successfully")
    message: Optional[str] = Field(None, description="Optional message")
    total_tracks: Optional[int] = Field(None, description="Total number of tracks in the graph")
    total_relationships: Optional[int] = Field(None, description="Total number of relationships (edges) in the graph")
    player_count: Optional[int] = Field(None, description="Number of distinct players (connected components)")
    player_groups: Optional[List[List[int]]] = Field(None, description="List of player groups (track IDs grouped by player)")
    verification_mode: Optional[str] = Field(None, description="Current verification mode")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")