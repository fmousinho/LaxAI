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
    """Response model for frame navigation metadata.
    
    Note: This endpoint returns ONLY navigation/session metadata.
    For detection/annotation data, use the /frames/{session_id}/{frame_id}/recipe endpoint.
    This ensures a single source of truth for annotation data.
    """
    
    frame_id: int = Field(..., description="The current frame index")
    video_id: str = Field(..., description="Video identifier")
    session_id: str = Field(..., description="Session identifier")
    has_next_frame: bool = Field(..., description="Whether there are more frames to read")
    has_previous_frame: bool = Field(..., description="Whether there are previous frames")
    total_frames: int = Field(..., description="Total number of frames in the video")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "frame_id": 150,
                "video_id": "test_video",
                "session_id": "12345678-01",
                "has_next_frame": True,
                "has_previous_frame": True,
                "total_frames": 300
            }
        }


class AnnotationDataResponse(BaseModel):
    """Response model for frame annotation data (detections + rendering config).
    
    This is the single source of truth for detection and annotation data.
    Contains supervision.Detections serialized to JSON plus lightweight rendering configuration.
    """
    
    frame_id: int = Field(..., description="Frame identifier")
    video_id: str = Field(..., description="Video identifier")
    session_id: str = Field(..., description="Session identifier")
    detections: Dict[str, Any] = Field(..., description="Serialized supervision.Detections object")
    rendering_config: Dict[str, Any] = Field(..., description="Rendering/style configuration")
    has_next: bool = Field(..., description="Whether there are more frames")
    has_previous: bool = Field(..., description="Whether there are previous frames")
    total_frames: int = Field(..., description="Total number of frames in the video")
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "frame_id": 150,
                "video_id": "test_video",
                "session_id": "12345678-01",
                "detections": {
                    "xyxy": [[100.0, 150.0, 200.0, 350.0]],
                    "confidence": [0.95],
                    "tracker_id": [42],
                    "class_id": [0],
                    "data": {
                        "player_id": [5]
                    }
                },
                "rendering_config": {
                    "player_styles": {
                        "5": {"preset": "default", "color": "#FF5733"}
                    },
                    "tracker_styles": {},
                    "custom_colors": {}
                },
                "has_next": True,
                "has_previous": True,
                "total_frames": 300
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

# Example data constants to avoid repetition
PLAYER_LIST_ITEM_EXAMPLE = {
    "player_id": 5,
    "tracker_ids": [42],
    "player_name": "John Doe",
    "image_path": "/path/to/player_image.png",
    "player_number": 23
}

class PlayerListItem(BaseModel):
    """Model representing a single player in the session."""
    player_id: int = Field(..., description="Unique player ID")
    tracker_ids: List[int] = Field(..., description="Tracker IDs associated with the player")
    player_name: Optional[str] = Field(..., description="Name of the player")
    image_path: Optional[str] = Field(..., description="Path to the player's image")
    player_number: Optional[int] = Field(None, description="Jersey number of the player")
    team_id: Optional[int] = Field(None, description="Team identifier for the player")

    class ConfigDict:
        json_schema_extra = {
            "example": PLAYER_LIST_ITEM_EXAMPLE
        }

class PlayerCreate(BaseModel):
    """Model for creating a new player in the session."""
    player_name: Optional[str] = Field(None, description="Name of the player to add")
    tracker_ids: List[int] = Field(..., description="List of tracker IDs to associate with the player")
    image_path: Optional[str] = Field(None, description="Path to the player's image")
    player_number: Optional[int] = Field(None, description="Jersey number of the player")
    team_id: Optional[int] = Field(None, description="Team identifier for the player")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "player_name": "John Doe",
                "tracker_ids": [42],
                "image_path": "/path/to/player_image.png"
            }
        }


class GetPlayersResponse(BaseModel):
    """Model representing a list of players identified in the session."""

    players: List[PlayerListItem] = Field(..., description="List of players in the session")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "players": [PLAYER_LIST_ITEM_EXAMPLE]
            }
        }

class PatchUpdatePlayerRequest(BaseModel):
    """Model for updating an existing player's information in the session."""
    player_id: int = Field(..., description="Unique player ID to update")
    player_name: Optional[str] = Field(None, description="Updated name of the player")
    tracker_ids: Optional[List[int]] = Field(None, description="Updated list of tracker IDs associated with the player")
    image_path: Optional[str] = Field(None, description="Updated path to the player's image")
    player_number: Optional[int] = Field(None, description="Updated jersey number of the player")
    team_id: Optional[int] = Field(None, description="Updated team identifier of the player")

    class ConfigDict:
        json_schema_extra = {
            "example": PLAYER_LIST_ITEM_EXAMPLE
        }
