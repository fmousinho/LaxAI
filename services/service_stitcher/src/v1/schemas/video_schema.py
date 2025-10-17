from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
from enum import Enum


class ImageFormat(str, Enum):
    """Supported image formats for frame streaming."""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"


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
    total_frames: int = Field(..., description="Total number of frames in the video")
    has_next_frame: bool = Field(..., description="Indicates if there is a next frame available")
    has_previous_frame: bool = Field(..., description="Indicates if there is a previous frame available")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "session_id": "12345678-01",
                "total_frames": 1500,
                "has_next_frame": True,
                "has_previous_frame": False,
            }
        }


class DetectionInfo(BaseModel):
    """Detection information for a single object in a frame."""
    
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    tracker_id: Optional[int] = Field(None, description="Tracker ID from detection/tracking")
    player_id: Optional[int] = Field(None, description="Mapped player ID")
    confidence: float = Field(..., description="Detection confidence score (0.0-1.0)")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "bbox": [100.0, 150.0, 200.0, 350.0],
                "tracker_id": 42,
                "player_id": 5,
                "confidence": 0.95
            }
        }


class FrameMetadataResponse(BaseModel):
    """Response model for frame metadata (for client-side annotation)."""
    
    frame_id: int = Field(..., description="The current frame index")
    video_id: str = Field(..., description="Video identifier")
    session_id: str = Field(..., description="Session identifier")
    detections: List[DetectionInfo] = Field(default_factory=list, description="Detection data for the frame")
    player_mappings: Dict[int, int] = Field(default_factory=dict, description="Tracker ID to Player ID mappings")
    has_next_frame: bool = Field(..., description="Whether there are more frames to read")
    has_previous_frame: bool = Field(..., description="Whether there are previous frames")
    total_frames: int = Field(..., description="Total number of frames in the video")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "frame_id": 150,
                "video_id": "test_video",
                "session_id": "12345678-01",
                "detections": [
                    {
                        "bbox": [100.0, 150.0, 200.0, 350.0],
                        "tracker_id": 42,
                        "player_id": 5,
                        "confidence": 0.95
                    }
                ],
                "player_mappings": {42: 5, 43: 7},
                "has_next_frame": True,
                "has_previous_frame": True,
                "total_frames": 300
            }
        }


class AnnotationRecipeResponse(BaseModel):
    """Response model for annotation recipe."""
    
    frame_id: int = Field(..., description="Frame identifier")
    recipe: Dict = Field(..., description="Declarative annotation recipe")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "frame_id": 150,
                "recipe": {
                    "frame_id": 150,
                    "instructions": [
                        {
                            "type": "bbox",
                            "coords": [100.0, 150.0, 200.0, 350.0],
                            "player_id": 5,
                            "tracker_id": 42,
                            "style_preset": "default",
                            "confidence": 0.95
                        }
                    ],
                    "metadata": {}
                }
            }
        }


class VideoGenerationRequest(BaseModel):
    """Request model for generating annotated video with user edits."""
    
    frame_recipes: Dict[int, Dict] = Field(
        ..., 
        description="Dictionary mapping frame_id to annotation recipe"
    )
    output_format: str = Field(
        default="mp4",
        description="Output video format"
    )
    fps: int = Field(
        default=30,
        description="Frames per second for output video"
    )
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "frame_recipes": {
                    "0": {
                        "frame_id": 0,
                        "instructions": [
                            {
                                "type": "bbox",
                                "coords": [100.0, 150.0, 200.0, 350.0],
                                "player_id": 7,  # User changed from 5 to 7
                                "tracker_id": 42,
                                "style_preset": "default"
                            }
                        ]
                    }
                },
                "output_format": "mp4",
                "fps": 30
            }
        }


class VideoGenerationResponse(BaseModel):
    """Response model for video generation task."""
    
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status (queued, processing, completed, failed)")
    message: Optional[str] = Field(None, description="Status message")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456",
                "status": "queued",
                "message": "Video generation task queued successfully"
            }
        }


class VideoFrameResponse(BaseModel):
    """Response model for frame operations (deprecated - kept for compatibility)."""

    frame_id: int = Field(..., description="The current frame index")
    frame_data: Any = Field(..., description="Frame image data (numpy array format)")
    has_next_frame: bool = Field(..., description="Whether there are more frames to read")
    has_previous_frame: bool = Field(..., description="Whether there are previous frames")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "frame_id": 30,
                "frame_data": "numpy array",
                "has_next_frame": True,
                "has_previous_frame": True,
            }
        }