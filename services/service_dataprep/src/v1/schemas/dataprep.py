"""
DataPrep API schemas for LaxAI DataPrep Service.

Complete schemas for API request/response validation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessFoldersResponse(BaseModel):
    """Response model for listing process folders."""

    folders: List[str] = Field(..., description="List of available process folder names (video IDs)")


class StartPrepRequest(BaseModel):
    """Request model for starting a verification session."""

    process_folder: str = Field(..., description="Process folder name (video ID) to start verification for")

    class Config:
        json_schema_extra = {
            "example": {
                "process_folder": "video_001"
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
    group1_id: Optional[int] = Field(None, description="First group ID for verification")
    group2_id: Optional[int] = Field(None, description="Second group ID for verification")
    group1_prefixes: Optional[List[str]] = Field(None, description="GCS prefixes for group1 tracks")
    group2_prefixes: Optional[List[str]] = Field(None, description="GCS prefixes for group2 tracks")
    total_pairs: Optional[int] = Field(None, description="Total number of possible pairs")
    verified_pairs: Optional[int] = Field(None, description="Number of pairs already verified")


class RecordResponseRequest(BaseModel):
    """Request model for recording a user response."""

    decision: str = Field(..., description="User's decision: 'same', 'different', or 'skip'")

    class Config:
        json_schema_extra = {
            "example": {
                "decision": "same"
            }
        }


class RecordResponseResponse(BaseModel):
    """Response model for recording a user response."""

    success: bool = Field(..., description="Whether the response was recorded successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class SaveGraphResponse(BaseModel):
    """Response model for saving the graph."""

    success: bool = Field(..., description="Whether the graph was saved successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class SaveGraphImageResponse(BaseModel):
    """Response model for saving the graph image."""

    success: bool = Field(..., description="Whether the graph image was saved successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class SuspendPrepResponse(BaseModel):
    """Response model for suspending a verification session."""

    success: bool = Field(..., description="Whether the session was suspended successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class MoveCropsResponse(BaseModel):
    """Response model for moving crops to verified tracks."""

    success: bool = Field(..., description="Whether all crops were moved successfully")
    message: Optional[str] = Field(None, description="Optional message about the operation")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")