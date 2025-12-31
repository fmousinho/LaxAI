from typing import Any, Dict, List, Optional, Union
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Calculate root directory for .env location
# Path: services/service_tracking/src/tracker/config.py -> services/service_tracking/src/tracker -> services/service_tracking/src -> services/service_tracking -> services -> root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
env_path = os.path.join(root_dir, ".env")


class DetectionModelConfig(BaseSettings):
    """Configuration for the detection model, including WandB artifact settings."""

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    # WandB settings
    wandb_api_key: str = Field(
        ...,
        description="WandB API key for authentication. Used to download the detection model from WandB.",
    )
    wandb_model_artifact: str = Field(
        default="fmousinho76-home-org/wandb-registry-model/Detections:latest",
        description="WandB artifact path for the detection model",
    )
    wandb_project: str = Field(
        default="LaxAI-Tracking",
        description="WandB project name for tracking model downloads. Used by WandB to group runs.",
    )
    wandb_run_name: str = Field(
        default="model-download",
        description="WandB run name for model download operations.",
    )
    # Model file settings
    artifact_file_name: str = Field(
        default="common-models-detection_latest.pth",
        description="Filename of the model weights within the artifact.",
    )
    num_classes: int = Field(
        default=6,
        description="Number of classes the model was trained with. If not provided, it is automatically detected.",
    )
    # Device settings
    device: Optional[str] = Field(
        default=None,
        description="Device to run the model on (cuda/mps/cpu). Auto-detected if None.",
    )

# Global detection model config instance
detection_model_config = DetectionModelConfig()


class TrackingParams(BaseSettings):
    """Tracking hyperparameters - mirrors DetectionConfig and TrackerConfig."""

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Detection Params
    default_confidence_threshold: float = Field(
        default=0.4,
        description="Default detection confidence threshold. Detections with lower values are discarded.",
    )
    nms_iou_threshold: float = Field(
        default=0.3,
        description="Detections that overlap (IoU) with each other by more than this threshold are discarded.",
    )
    border_margin: int = Field(
        default=2,
        description="Detections with bounding box less than border_margin from the frame's edges are discarded.",
    )

    # Tracking Params
    prediction_threshold: float = Field(
        default=0.4, 
        description="Minimum detection confidence for it to be considered for tracking."
    )
    track_activation_threshold: float = Field(
        default=0.7, 
        description="Used to separate high and low confidence detections, per BYTETrack algorithm."
    )
    lost_track_buffer: int = Field(
        default=3, 
        description="How many frames to wait for a lost track to be reactivated before removing it."
    )
    high_conf_max_distance: float = Field(
        default=0.8, 
        description="Max distance for tracks to be matched when using high confidence detections."
    )
    low_conf_max_distance: float = Field(
        default=0.5, 
        description="Max distance for tracks to be matched when using low confidence detections."
    )
    unconfirmed_max_distance: float = Field(
        default=0.8, 
        description="Max distance for tracks to be matched when using unconfirmed detections."
    )
    min_consecutive_frames: int = Field(
        default=3, 
        description="Minimum consecutive frames for tracks to be confirmed."
    )
    use_only_confirmed_tracks: bool = Field(
        default=True,
        description="If True, only confirmed tracks are written to the output. If False, all tracks are returned."
    )
    
    # Embedding Config
    enable_reid: bool = Field(
        default=True,
        description="Enable ReID feature extraction during tracking. Required for player association later."
    )
    wandb_api_key: Optional[str] = Field(
        default=None,
        description="Optional WandB API key to override environment variable."
    )
    embedding_update_frequency: int = Field(
        default=5, 
        description="Frequency (in frames) to update track embeddings list."
    )
    embedding_quality_threshold: float = Field(
        default=0.6, 
        description="Minimum detection score to trigger embedding update."
    )

    # Cost Matrix Config
    apply_aspect_ratio_penalty: bool = Field(
        default=True,
        description="Apply aspect ratio change penalty to cost matrix."
    )
    apply_height_gate: bool = Field(
        default=True,
        description="Apply height change penalty to cost matrix."
    )
    apply_enforce_min_distance: bool = Field(
        default=True,
        description="Rejects associations if the best distance is less than enforce_min_distance_threshold from the second best distance."
    )
    aspect_ratio_factor: float = Field(
        default=0.6,
        description="Factor to apply to aspect ratio change penalty."
    )
    height_threshold: float = Field(
        default=0.2,
        description="Maximum height change for association to be accepted."
    )
    min_distance_threshold: float = Field(
        default=0.25,
        description="Minimum separation for association to be accepted."
    )


class KalmanFilterConfig(BaseSettings):
    """Configuration for the Kalman Filter used in tracking."""

    std_weight_position: float = Field(
        default=0.198,  # Obtained from actual video analysis
        description="Standard deviation weight for position",
    )
    std_weight_velocity: float = Field(
        default=0.014695, # Obtained from actual video analysis
        description="Standard deviation weight for velocity",
    )

kalman_filter_config = KalmanFilterConfig()


