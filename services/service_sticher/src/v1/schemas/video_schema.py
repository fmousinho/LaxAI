from pydantic import BaseModel, Field
from typing import Any


class VideoLoadRequest(BaseModel):
    """Request model for loading a video."""

    tenant_id: str = Field(..., description="Unique identifier for the tenant")
    video_path: str = Field(..., description="Path or URL of the video to load in GCS")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "tenant_id": "tenant_name",
                "video_path": "/path/to/video.mp4"
            }
        }

class VideoLoadResponse(BaseModel):
    """Response model for loading a video."""

    session_id: str = Field(..., description="Unique identifier for the video session in the format TTTTTTTT-RR")
    video_path: str = Field(..., description="Path or URL of the loaded video")
    total_frames: int = Field(..., description="Total number of frames in the video")
    has_next_frame: bool = Field(..., description="Indicates if there is a next frame available")
    has_previous_frame: bool = Field(..., description="Indicates if there is a previous frame available")

    class ConfigDict:
        schema_extra = {
            "example": {
                "session_id": "12345678-01",
                "video_path": "/path/to/video.mp4",
                "total_frames": 1500,
                "has_next_frame": True,
                "has_previous_frame": False,
            }
        }


class VideoFrameResponse(BaseModel):
    """Response model for frame operations."""

    frame_id: int = Field(..., description="The current frame index")
    frame_data: Any = Field(..., description="Frame image data (numpy array format)")
    has_next_frame: bool = Field(..., description="Whether there are more frames to read")
    has_previous_frame: bool = Field(..., description="Whether there are previous frames")

    class ConfigDict:
        schema_extra = {
            "example": {
                "frame_id": 30,
                "frame_data": "numpy array",
                "has_next_frame": True,
                "has_previous_frame": True,
            }
        }