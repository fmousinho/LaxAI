"""
Configuration classes for training and inference.
Centralized location for all hyperparameters and configuration settings.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import sys

@dataclass
class DebugConfig:
    """Configuration for debugging and logging."""
    bypass_player_creation: bool = True
    save_detections_file: Optional[str] = "tracks.json"  # Path to save detections JSON file (None = disabled)



@dataclass
class ModelConfig:
    """Configuration for model dimensions and architecture."""
    input_height: int = 120
    input_width: int = 80
    embedding_dim: int = 512
    dropout_rate: float = 0.5  # Dropout rate for regularization
    enable_grass_mask: bool = False  # Not fully implemented yet

    # ImageNet normalization values (for pretrained ResNet backbone)
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrackerConfig:
    """Configuration for ByteTrack and tracking parameters."""
    track_activation_threshold: float = 0.7
    lost_track_buffer: int = 5
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 10
    crop_save_interval: int = 5
    id_type: str = 'external'  # Type of ID to use ('internal' or 'external')
    # Velocity transformation parameters
    transform_velocities: bool = True  # Whether to transform velocities with affine matrix
    scale_height_velocity: bool = True  # Whether to scale height velocity based on scaling factor
    scaling_threshold: float = 0.1  # Minimum scale factor change to trigger height velocity scaling


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 15
    margin: float = 0.5
    weight_decay: float = 1e-4  # L2 regularization weight decay
    model_save_path: str = '/common/model/embeddings_latest.pth'
    train_ratio: float = 0.8
    min_images_per_player: int = 3
    num_workers: int = 4 if sys.platform != "darwin" else 0  # Number of DataLoader workers
    early_stopping_loss_ratio: float = 0.1  # Early stopping threshold as a ratio of margin
    early_stopping_patience: Optional[int] = 5  # Number of epochs to wait before early stopping (None = disabled)


@dataclass
class DetectionConfig:
    """Configuration for detection and processing parameters."""
    nms_iou_threshold: Optional[float] = None
    player_class_id: int = 3
    prediction_threshold: float = 0.6
    model_checkpoint: str = "detection_latest.pth"
    output_video_path: str = "results.mp4"
    crop_extract_interval: int = 5
    # Google Storage configuration
    checkpoint_dir: str = "models"  # Path for model storage within the 'common' GCS directory
    # Color space handling
    color_space: str = "RGB"  # Expected color space for processing
    convert_bgr_to_rgb: bool = True  # Auto-convert OpenCV BGR to RGB
    # Training pipeline configuration
    delete_original_raw_videos: bool = False  # Whether to delete original raw video files after processing
    frames_per_video: int = 20  # Number of frames to extract per video


@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters."""
    batch_size: int = 128
    num_workers: int = 4 if sys.platform != "darwin" else 0  
    dbscan_eps: float = 0.1
    dbscan_min_samples: int = 5
    # Target cluster range for adaptive search
    target_min_clusters: int = 20
    target_max_clusters: int = 40
    # Eps search parameters
    initial_eps: float = 0.1
    max_eps: float = 0.9
    min_eps: float = 0.01
    eps_adjustment_factor: float = 0.2
    max_eps_searches: int = 20


@dataclass
class PlayerConfig:
    """Configuration for player association parameters."""
    reid_similarity_threshold: float = 0.9


@dataclass
class TrackStitchingConfig:
    """Configuration for track stitching parameters."""
    stich_tracks_after_tracker: bool = False
    enable_stitching: bool = False
    stitch_similarity_threshold: float = 0.9
    max_time_gap: int = 60  # Maximum frame gap between tracklets
    appearance_weight: float = 1.0
    temporal_weight: float = .5
    motion_weight: float = 0.1  # Future use for motion-based cost


@dataclass
class TransformConfig:
    """Configuration for data augmentation and transforms."""
    # Data augmentation parameters
    hflip_prob: float = 0.5
    colorjitter_brightness: float = 0.2
    colorjitter_contrast: float = 0.2
    colorjitter_saturation: float = 0.2
    colorjitter_hue: float = 0.1
    random_rotation_degrees: int = 10
    random_affine_degrees: int = 0
    random_affine_translate: Tuple[float, float] = (0.1, 0.1)
    
    # Background removal configuration
    enable_background_removal: bool = True  # Global flag to enable/disable background removal
    background_detector_sample_frames: int = 5  # Frames to use for background detector training
    background_detector_auto_train: bool = True  # Whether to auto-train detector from sample images


@dataclass
class BackgroundMaskConfig:
    """Configuration for background mask detection and removal."""
    # Background detection parameters
    sample_frames: int = 5  # Number of frames to sample for background detection
    std_dev_multiplier: float = 1.0  # Number of standard deviations for color bounds
    replacement_color: Tuple[int, int, int] = (255, 255, 255)  # RGB color to replace background with (white)
    verbose: bool = True  # Whether to print progress information
    
    # Frame processing parameters
    top_crop_ratio: float = 0  # Remove top 0% of frame
    bottom_crop_ratio: float = 0  # Remove bottom 0% of frame

    # HSV color space limits
    hsv_min_values: Tuple[int, int, int] = (0, 0, 0)  # Minimum HSV values
    hsv_max_values: Tuple[int, int, int] = (179, 255, 255)  # Maximum HSV values
    
    # Default bounds adjustment settings
    default_std_multiplier: float = 1.0  # Default std deviation multiplier
    min_std_multiplier: float = 0.1  # Minimum allowed std deviation multiplier
    max_std_multiplier: float = 3.0  # Maximum allowed std deviation multiplier


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "LaxAI"
    entity: str = "fmousinho76"
    team: str = "fmousinho76-home"
    tags: List[str] = field(default_factory=lambda: ["dev"])
    log_frequency: int = 10
    save_model_artifacts: bool = True
    log_sample_images: bool = True
    sample_images_count: int = 20
    log_all_images: bool = True
    model_name: str = "siamese-net-embeddings"
    


# Global config instances - these can be imported and used directly
model_config = ModelConfig()
tracker_config = TrackerConfig()
training_config = TrainingConfig()
detection_config = DetectionConfig()
clustering_config = ClusteringConfig()
player_config = PlayerConfig()
track_stitching_config = TrackStitchingConfig()
transform_config = TransformConfig()
background_mask_config = BackgroundMaskConfig()
debug_config = DebugConfig()
wandb_config = WandbConfig()
