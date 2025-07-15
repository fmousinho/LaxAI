"""
Image transforms for training and inference.
Contains all image preprocessing pipelines and transform functions.
"""

import torchvision.transforms as transforms
import cv2
import numpy as np
from .all_config import model_config, transform_config, detection_config, background_mask_config
from typing import Optional, List
from PIL import Image


# Global background detector instance
_global_background_detector: Optional['BackgroundMaskDetector'] = None


def set_global_background_detector(detector: Optional['BackgroundMaskDetector']):
    """
    Set the global background detector instance.
    
    Args:
        detector: BackgroundMaskDetector instance or None to disable
    """
    global _global_background_detector
    _global_background_detector = detector


def get_global_background_detector() -> Optional['BackgroundMaskDetector']:
    """
    Get the current global background detector instance.
    
    Returns:
        BackgroundMaskDetector instance or None if not set
    """
    return _global_background_detector


def create_background_detector_from_images(images: List[np.ndarray]) -> 'BackgroundMaskDetector':
    """
    Create and train a background detector from a list of sample images.
    
    Args:
        images: List of RGB images to use for background detection
        
    Returns:
        Trained BackgroundMaskDetector instance
    """
    # Import here to avoid circular imports
    from modules.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
    
    # Create frame generator
    frame_generator = create_frame_generator_from_images(images)
    
    # Create detector with configuration
    detector = BackgroundMaskDetector(
        frame_generator=frame_generator,
        config=background_mask_config,
        sample_frames=transform_config.background_detector_sample_frames,
        verbose=False
    )
    
    return detector


def initialize_background_removal(sample_images: Optional[List[np.ndarray]] = None):
    """
    Initialize background removal system based on configuration.
    
    Args:
        sample_images: Optional list of sample images for background detector training.
                      If None and auto_train is True, will need to be set later.
    """
    global _global_background_detector
    
    if not transform_config.enable_background_removal:
        _global_background_detector = None
        return
    
    if sample_images is not None and transform_config.background_detector_auto_train:
        _global_background_detector = create_background_detector_from_images(sample_images)
    else:
        # Background removal is enabled but no detector set yet
        # This can be set later using set_global_background_detector
        _global_background_detector = None


def is_background_removal_enabled() -> bool:
    """
    Check if background removal is enabled and available.
    
    Returns:
        True if background removal is enabled and detector is available
    """
    return (transform_config.enable_background_removal and 
            _global_background_detector is not None)


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


class BackgroundRemovalTransform:
    """
    Transform class for applying background removal to images.
    
    This transform uses the global background detector configuration to apply
    background removal when enabled.
    """
    
    def __init__(self, override_detector: Optional['BackgroundMaskDetector'] = None):
        """
        Initialize the background removal transform.
        
        Args:
            override_detector: Optional detector to override the global one.
                              If None, uses the global detector configuration.
        """
        self.override_detector = override_detector
    
    def __call__(self, image):
        """
        Apply background removal to the image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Processed image with background removed (same type as input)
        """
        # Determine which detector to use
        detector = self.override_detector if self.override_detector is not None else _global_background_detector
        
        # Skip if background removal is disabled or no detector available
        if not transform_config.enable_background_removal or detector is None:
            return image
        
        # Convert PIL to numpy if needed
        was_pil = isinstance(image, Image.Image)
        if was_pil:
            image_array = np.array(image)
        else:
            image_array = image
        
        # Ensure RGB format
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = ensure_rgb_format(image_array, "RGB")
        
        # Apply background removal
        processed_image = detector.remove_background(image_array)
        
        # Convert back to PIL if input was PIL
        if was_pil:
            return Image.fromarray(processed_image)
        else:
            return processed_image


def create_background_removal_transform(override_detector: Optional['BackgroundMaskDetector'] = None):
    """
    Create a background removal transform.
    
    Args:
        override_detector: Optional detector to override the global configuration
        
    Returns:
        BackgroundRemovalTransform instance
    """
    return BackgroundRemovalTransform(override_detector)


def create_training_transforms():
    """Create training transforms with data augmentation."""
    transforms_list = []
    
    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())
    
    # Add standard transforms
    transforms_list.extend([
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
    
    return transforms.Compose(transforms_list)


def create_inference_transforms():
    """Create inference transforms without augmentation."""
    transforms_list = []
    
    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())
    
    # Add standard transforms
    transforms_list.extend([
        transforms.Resize((model_config.input_height, model_config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std)
    ])
    
    return transforms.Compose(transforms_list)


def create_tensor_to_pil_transforms():
    """Create transforms for converting tensor back to PIL Image."""
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in model_config.imagenet_std]),
        transforms.Normalize(mean=[-m for m in model_config.imagenet_mean], std=[1., 1., 1.]),
        transforms.ToPILImage()
    ])


def create_opencv_safe_transforms():
    """Create transforms that handle OpenCV BGR input safely."""
    transforms_list = [
        transforms.Lambda(lambda x: ensure_rgb_format(x, "BGR") if isinstance(x, np.ndarray) else x),
        transforms.ToPILImage(),
    ]
    
    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())
    
    # Add standard transforms
    transforms_list.extend([
        transforms.Resize((model_config.input_height, model_config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std)
    ])
    
    return transforms.Compose(transforms_list)


def create_opencv_safe_training_transforms():
    """Create training transforms that handle OpenCV BGR input safely."""
    transforms_list = [
        transforms.Lambda(lambda x: ensure_rgb_format(x, "BGR") if isinstance(x, np.ndarray) else x),
        transforms.ToPILImage(),
    ]
    
    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())
    
    # Add standard transforms
    transforms_list.extend([
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
    
    return transforms.Compose(transforms_list)


def refresh_transform_instances():
    """
    Refresh all transform instances to pick up configuration changes.
    Call this after modifying transform configuration or background detector.
    """
    global training_transforms, inference_transforms, validation_transforms
    global opencv_safe_transforms, opencv_safe_training_transforms, TRANSFORMS
    
    # Recreate all transform instances
    training_transforms = create_training_transforms()
    inference_transforms = create_inference_transforms()
    validation_transforms = inference_transforms  # Same as inference
    opencv_safe_transforms = create_opencv_safe_transforms()
    opencv_safe_training_transforms = create_opencv_safe_training_transforms()
    
    # Update the transforms dictionary
    TRANSFORMS.update({
        'training': training_transforms,
        'inference': inference_transforms,
        'validation': validation_transforms,
        'opencv_safe': opencv_safe_transforms,
        'opencv_safe_training': opencv_safe_training_transforms
    })


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


def create_transforms_with_background_removal(background_detector: 'BackgroundMaskDetector'):
    """
    Create all transform variants with a specific background detector.
    
    This function is deprecated. Use set_global_background_detector() and 
    refresh_transform_instances() instead.
    
    Args:
        background_detector: Pre-trained BackgroundMaskDetector instance
        
    Returns:
        dict: Dictionary of transform pipelines with background removal
    """
    # Temporarily set the global detector
    old_detector = get_global_background_detector()
    old_enabled = transform_config.enable_background_removal
    
    try:
        # Enable background removal and set detector
        transform_config.enable_background_removal = True
        set_global_background_detector(background_detector)
        
        # Create transforms with background removal
        result = {
            'training': create_training_transforms(),
            'inference': create_inference_transforms(),
            'validation': create_inference_transforms(),
            'tensor_to_pil': tensor_to_pil,  # No background removal needed for tensor->PIL
            'opencv_safe': create_opencv_safe_transforms(),
            'opencv_safe_training': create_opencv_safe_training_transforms()
        }
        
        return result
        
    finally:
        # Restore original settings
        transform_config.enable_background_removal = old_enabled
        set_global_background_detector(old_detector)


def get_transforms(mode='inference', background_detector: Optional['BackgroundMaskDetector'] = None):
    """
    Get transforms for the specified mode.
    
    Args:
        mode (str): One of 'training', 'inference', 'validation', 'tensor_to_pil', 'opencv_safe', 'opencv_safe_training'
        background_detector: Deprecated. Use set_global_background_detector() instead.
        
    Returns:
        transforms.Compose: The requested transform pipeline
        
    Raises:
        ValueError: If mode is not recognized
    """
    if background_detector is not None:
        # Legacy support - use the old approach
        bg_transforms = create_transforms_with_background_removal(background_detector)
        if mode not in bg_transforms:
            raise ValueError(f"Unknown transform mode: {mode}. Available modes: {list(bg_transforms.keys())}")
        return bg_transforms[mode]
    else:
        # Use the new centralized approach
        if mode not in TRANSFORMS:
            raise ValueError(f"Unknown transform mode: {mode}. Available modes: {list(TRANSFORMS.keys())}")
        return TRANSFORMS[mode]
