"""
Image transforms for training and inference.
Contains all image preprocessing pipelines and transform functions.
"""

import torchvision.transforms as transforms
import cv2
import numpy as np
from .all_config import model_config, transform_config, detection_config


def ensure_rgb_format(image_array: np.ndarray, source_format: str = "BGR") -> np.ndarray:
    """
    Ensure image is in RGB format for consistent processing.
    
    Args:
        image_array: Input image array
        source_format: Source format ("BGR" or "RGB")
        
    Returns:
        Image array in RGB format
    """
    if detection_config.convert_bgr_to_rgb and source_format == "BGR":
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_array


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


def create_opencv_safe_transforms():
    """Create transforms that handle OpenCV BGR input safely."""
    return transforms.Compose([
        transforms.Lambda(lambda x: ensure_rgb_format(x, "BGR") if isinstance(x, np.ndarray) else x),
        transforms.ToPILImage(),
        transforms.Resize((model_config.input_height, model_config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std)
    ])


def create_opencv_safe_training_transforms():
    """Create training transforms that handle OpenCV BGR input safely."""
    return transforms.Compose([
        transforms.Lambda(lambda x: ensure_rgb_format(x, "BGR") if isinstance(x, np.ndarray) else x),
        transforms.ToPILImage(),
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


# Create transform instances
training_transforms = create_training_transforms()
inference_transforms = create_inference_transforms()
validation_transforms = inference_transforms  # Same as inference
tensor_to_pil = create_tensor_to_pil_transforms()
opencv_safe_transforms = create_opencv_safe_transforms()
opencv_safe_training_transforms = create_opencv_safe_training_transforms()

# Dictionary for easy access to all transforms
TRANSFORMS = {
    'training': training_transforms,
    'inference': inference_transforms,
    'validation': validation_transforms,
    'tensor_to_pil': tensor_to_pil,
    'opencv_safe': opencv_safe_transforms,
    'opencv_safe_training': opencv_safe_training_transforms
}


def get_transforms(mode='inference'):
    """
    Get transforms for the specified mode.
    
    Args:
        mode (str): One of 'training', 'inference', 'validation', 'tensor_to_pil', 'opencv_safe', 'opencv_safe_training'
        
    Returns:
        transforms.Compose: The requested transform pipeline
        
    Raises:
        ValueError: If mode is not recognized
    """
    if mode not in TRANSFORMS:
        raise ValueError(f"Unknown transform mode: {mode}. Available modes: {list(TRANSFORMS.keys())}")
    
    return TRANSFORMS[mode]
