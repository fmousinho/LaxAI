"""
Transform configurations for training and inference.
Centralized location for all image preprocessing pipelines and module configurations.
"""

import torchvision.transforms as transforms
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import sys


@dataclass
class ModelConfig:
    """Configuration for model dimensions and architecture."""
    input_height: int = 120
    input_width: int = 80
    embedding_dim: int = 512
    
    # ImageNet normalization values (for pretrained ResNet backbone)
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrackerConfig:
    """Configuration for ByteTrack and tracking parameters."""
    track_activation_threshold: float = 0.5
    lost_track_buffer: int = 5
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 30
    crop_save_interval: int = 5


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 20
    margin: float = 0.1
    model_save_path: str = 'lacrosse_reid_model.pth'
    train_ratio: float = 0.8
    min_images_per_player: int = 3
    num_workers: int = 4 if sys.platform != "darwin" else 0  # Number of DataLoader workers
    early_stopping_loss_ratio: float = 0.1  # Early stopping threshold as a ratio of margin


@dataclass
class DetectionConfig:
    """Configuration for detection and processing parameters."""
    nms_iou_threshold: Optional[float] = None
    player_class_id: int = 3
    prediction_threshold: float = 0.6
    model_checkpoint: str = "checkpoint.pth"
    checkpoint_dir: str = "Colab_Notebooks"
    output_video_path: str = "results.mp4"
    crop_extract_interval: int = 5


@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters."""
    batch_size: int = 128
    num_workers: int = 4  # Number of DataLoader workers for clustering
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
    enable_stitching: bool = True
    stitch_similarity_threshold: float = 0.85
    max_time_gap: int = 60  # Maximum frame gap between tracklets
    appearance_weight: float = 1.0
    temporal_weight: float = 0.5
    motion_weight: float = 0.05


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


# Global config instances - these can be imported and used directly
model_config = ModelConfig()
tracker_config = TrackerConfig()
training_config = TrainingConfig()
detection_config = DetectionConfig()
clustering_config = ClusteringConfig()
player_config = PlayerConfig()
track_stitching_config = TrackStitchingConfig()
transform_config = TransformConfig()


def create_training_transforms():
    """Create training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((model_config.input_height, model_config.input_width)),
        transforms.RandomHorizontalFlip(p=transform_config.hflip_prob),
        transforms.ColorJitter(
            brightness=transform_config.colorjitter_brightness,
            contrast=transform_config.colorjitter_contrast,
            saturation=transform_config.colorjitter_saturation,
            hue=transform_config.colorjitter_hue
        ),
        transforms.RandomRotation(transform_config.random_rotation_degrees),
        transforms.RandomAffine(
            degrees=transform_config.random_affine_degrees,
            translate=transform_config.random_affine_translate
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std)
    ])


def create_inference_transforms():
    """Create inference transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize((model_config.input_height, model_config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std)
    ])


def create_tensor_to_pil_transforms():
    """Create transforms for converting tensor back to PIL Image."""
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in model_config.imagenet_std]),
        transforms.Normalize(mean=[-m for m in model_config.imagenet_mean], std=[1., 1., 1.]),
        transforms.ToPILImage()
    ])


# Create transform instances
training_transforms = create_training_transforms()
inference_transforms = create_inference_transforms()
validation_transforms = inference_transforms  # Same as inference
tensor_to_pil = create_tensor_to_pil_transforms()

# Dictionary for easy access to all transforms
TRANSFORMS = {
    'training': training_transforms,
    'inference': inference_transforms,
    'validation': validation_transforms,
    'tensor_to_pil': tensor_to_pil
}

def get_transforms(mode='inference'):
    """
    Get transforms for the specified mode.
    
    Args:
        mode (str): One of 'training', 'inference', 'validation', 'tensor_to_pil'
        
    Returns:
        transforms.Compose: The requested transform pipeline
        
    Raises:
        ValueError: If mode is not recognized
    """
    if mode not in TRANSFORMS:
        raise ValueError(f"Unknown transform mode: {mode}. Available modes: {list(TRANSFORMS.keys())}")
    
    return TRANSFORMS[mode]
