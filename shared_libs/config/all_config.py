"""
Configuration classes for training and inference.
Centralized location for all hyperparameters and configuration settings.
"""

import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple


def generate_unique_run_name() -> str:
    """Generate a unique run name with timestamp and short UUID"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"api_run_{timestamp}_{short_uuid}"


@dataclass
class GoogleStorageConfig:
    """Configuration for Google Cloud Storage."""

    project_id: str = "laxai-466119"
    gcs_paths_file: str = (
        "shared_libs/config/gcs_structure.yaml"  # Path to YAML file with GCS paths
    )
    bucket_name: str = "laxai_dev"
    credentials_name: str = "GOOGLE_APPLICATION_CREDENTIALS"


@dataclass
class DebugConfig:
    """Configuration for debugging and logging."""

    bypass_player_creation: bool = True
    # Path to save detections JSON file (None = disabled)
    save_detections_file: Optional[str] = "tracks.json"


@dataclass
class ModelConfig:
    """Configuration for model dimensions and architecture."""

    input_height: int = 224  # Increased to match DINOv3 expected input
    input_width: int = 224  # Increased to match DINOv3 expected input
    embedding_dim: int = 512  
    dropout_rate: float = 0.05  # Reduced dropout for fine-tuning
    resnet_conv_kernel_size: int = 3
    resnet_conv_stride: int = 1
    resnet_conv_padding: int = 1
    resnet_conv_bias: bool = False  # Whether to use bias in the first conv
    model_class_module: str = "siamesenet_dino"  # Module where the model class is defined
    model_class_str: str = "SiameseNet"  # Name of the model class

    # ImageNet normalization values (for pretrained ResNet backbone)
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrackerConfig:
    """Configuration for ByteTrack and tracking parameters."""

    track_activation_threshold: float = 0.7
    lost_track_buffer: int = 0
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 10
    crop_save_interval: int = 5
    id_type: str = "external"  # Type of ID to use ('internal' or 'external')
    # Velocity transformation parameters
    transform_velocities: bool = True  # Whether to transform velocities with affine matrix
    scale_height_velocity: bool = True  # Whether to scale height velocity based on scaling factor
    scaling_threshold: float = 0.1  # Minimum scale factor change to trigger height velocity scaling
    affine_median_translation_px: float = 0.75  # Median displacement below this -> treat as static
    affine_max_translation_px: float = 2.0  # Hard cap on max displacement to keep snapshot static
    affine_linear_fro_threshold: float = 0.015  # Frobenius norm delta from identity to skip warp


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    batch_size: int = 128  # Increased from 128 for better triplet mining
    num_workers: int = 8 if sys.platform != "darwin" else 0  # Number of DataLoader workers
    learning_rate: float = 1e-3  # Reduced from 1e-3 for fine-tuning
    lr_scheduler_patience: int = 5  # Increased patience
    lr_scheduler_threshold: float = 1e-3  # More sensitive threshold
    lr_scheduler_factor: float = 0.5  # More conservative reduction
    lr_scheduler_min_lr: float = 1e-6  # Lower minimum LR
    num_epochs: int = 100  # Increased epochs for fine-tuning
    margin: float = 0.3  # Increased from 0.2 for better separation
    weight_decay: float = 1e-5  # Reduced L2 regularization
    train_ratio: float = 0.8
    min_images_per_player: int = 3  # Increased minimum images per player
    force_pretraining: bool = False  # Force using ResNet defaults for pretraining
    early_stopping_patience: Optional[int] = 15  # More patience for convergence
    margin_decay_rate: float = 0.995  # Slower margin decay
    margin_change_threshold: float = 0.005  # More sensitive margin updates
    prefetch_factor: int = 4  # Increased prefetch for better GPU utilization
    
    # Classification head parameters for jump-starting embeddings
    use_classification_head: bool = False  # Whether to use classification head
    classification_epochs: int = 20  # Number of epochs to use classification loss
    classification_weight_start: float = 1.0  # Initial weight for classification loss
    
    # Number of datasets to use for training (0 = use all)
    n_datasets_to_use: Optional[int] = 0
    # Full GCS path to a specific dataset (gs://bucket/path or bucket/path)
    # If provided, this overrides dataset discovery and n_datasets_to_use
    dataset_address: Optional[str] = None
    # GPU memory management settings
    clear_memory_on_start: bool = True  # Clear GPU memory when training starts
    aggressive_memory_cleanup: bool = True  # Enable aggressive memory cleanup on errors
    
    # Auto-resume configuration for long-running Cloud Run jobs
    execution_timeout_minutes: int = 50  # Trigger auto-resume at this threshold (Cloud Run has 60min limit)


@dataclass
class EvaluatorConfig:
    """Configuration for model evaluation parameters."""

    threshold: float = (
        0.7  # Starting similarity threshold for evaluation (adjusted during evaluation)
    )
    number_of_workers: int = (
        0 if sys.platform != "darwin" else 0
    )  # Number of workers for DataLoader
    emb_batch_size: int = 32  # Batch size for embedding generation
    prefetch_factor: int = 2  # Number of batches to prefetch for DataLoader


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
    delete_original_raw_videos: bool = (
        True  # Whether to delete original raw video files after processing
    )
    frames_per_video: int = 3  # Number of frames to extract per video
    # Batch processing configuration
    batch_size: int = 32  # Number of frames to process in parallel (32 for GPU, 1 for CPU/MPS)
    # Note: This default assumes GPU availability. For CPU/MPS, this should be set to 1
    # at runtime based on the device being used.
    use_batch_detection: bool = True  # Enable batch detection for improved GPU utilization (default: enabled)
    # Set to False to fall back to sequential frame-by-frame detection for debugging


@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters."""

    batch_size: int = 128
    clustering_workers: int = 4 if sys.platform != "darwin" else 0
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
    temporal_weight: float = 0.5
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

    # Background removal configuration - we are using this for agumentation, not transforms
    enable_background_removal: bool = False  # Global flag to enable/disable background removal
    background_detector_sample_frames: int = 5  # Frames to use for background detector training
    background_detector_auto_train: bool = True  # Whether to auto-train detector from sample images


@dataclass
class BackgroundMaskConfig:
    """Configuration for background mask detection and removal."""

    # Background detection parameters
    sample_frames: int = 5  # Number of frames to sample for background detection
    std_dev_multiplier: float = 1.0  # Number of standard deviations for color bounds
    # RGB color to replace background with (white)
    replacement_color: Tuple[int, int, int] = (255, 255, 255)
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
    # model_name: str = "siamese-net-embeddings"
    # embeddings_model_name: str = "resnet-cbam-embeddings"
    detection_model_collection: str = "Detections"
    # embeddings_model_collection: str = "PlayerEmbeddings"
    default_model_versions_to_keep: int = 3
    model_tags_to_skip_deletion: List[str] = field(default_factory=lambda: ["do_not_delete"])
    run_name: str = "run"


@dataclass
class APIConfig:
    """Configuration for API endpoints and request handling."""

    verbose: bool = False  # Enable verbose logging for API requests
    resume_from_checkpoint: bool = True  # Resume from checkpoint if available
    # Default WandB tags for API requests
    default_wandb_tags: List[str] = field(default_factory=lambda: ["api"])

    @property
    def default_custom_name(self) -> str:
        """Generate a unique custom name for each API request"""
        return generate_unique_run_name()


# Global config instances - these can be imported and used directly
google_storage_config = GoogleStorageConfig()
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
evaluator_config = EvaluatorConfig()
api_config = APIConfig()
