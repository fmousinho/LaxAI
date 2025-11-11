"""
Image transforms for training and inference.
Contains all image preprocessing pipelines and transform functions.
"""

import random
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from .all_config import (background_mask_config, detection_config,
                         model_config, transform_config)

if TYPE_CHECKING:
    from common.background_mask import BackgroundMaskDetector


# Global background detector instance
_global_background_detector: Optional["BackgroundMaskDetector"] = None


def set_global_background_detector(detector: Optional["BackgroundMaskDetector"]):
    """
    Set the global background detector instance.

    Args:
        detector: BackgroundMaskDetector instance or None to disable
    """
    global _global_background_detector
    _global_background_detector = detector


def get_global_background_detector() -> Optional["BackgroundMaskDetector"]:
    """
    Get the current global background detector instance.

    Returns:
        BackgroundMaskDetector instance or None if not set
    """
    return _global_background_detector


def create_background_detector_from_images(images: List[np.ndarray]) -> "BackgroundMaskDetector":
    """
    Create and train a background detector from a list of sample images.

    Args:
        images: List of RGB images to use for background detection

    Returns:
        Trained BackgroundMaskDetector instance
    """
    # Import here to avoid circular imports
    from common.background_mask import (BackgroundMaskDetector,
                                        create_frame_generator_from_images)

    # Create frame generator
    frame_generator = create_frame_generator_from_images(images)

    # Create detector with configuration
    detector = BackgroundMaskDetector(
        frame_generator=frame_generator,
        config=background_mask_config,
        sample_frames=transform_config.background_detector_sample_frames,
        verbose=False,
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
    return transform_config.enable_background_removal and _global_background_detector is not None


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

    def __init__(self, override_detector: Optional["BackgroundMaskDetector"] = None):
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
        detector = (
            self.override_detector
            if self.override_detector is not None
            else _global_background_detector
        )

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
            # Ensure processed_image is a numpy array (should always be for single image input)
            if isinstance(processed_image, np.ndarray):
                return Image.fromarray(processed_image)
            else:
                raise ValueError("Background removal returned unexpected type")
        else:
            return processed_image


def create_background_removal_transform(
    override_detector: Optional["BackgroundMaskDetector"] = None,
):
    """
    Create a background removal transform.

    Args:
        override_detector: Optional detector to override the global configuration

    Returns:
        BackgroundRemovalTransform instance
    """
    return BackgroundRemovalTransform(override_detector)


# ============================================================================
# ADVANCED AUGMENTATION TRANSFORMS (replaces offline augmentation module)
# ============================================================================


class RandomRotationTransform:
    """
    Custom rotation transform that applies fixed angle rotations similar to augmentation module.
    """

    def __init__(self, angles: List[float] = [30, -30], prob: float = 0.5):
        """
        Args:
            angles: List of rotation angles to choose from
            prob: Probability of applying rotation
        """
        self.angles = angles
        self.prob = prob

    def __call__(self, image):
        if random.random() > self.prob:
            return image

        angle = random.choice(self.angles)

        if isinstance(image, Image.Image):
            return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        else:
            # Handle numpy arrays
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            return rotated


class StretchTransform:
    """
    Applies horizontal or vertical stretching followed by center cropping.
    """

    def __init__(self, stretch_factor: float = 2.0, prob: float = 0.3):
        """
        Args:
            stretch_factor: Factor to stretch by (2.0 = 100% stretch)
            prob: Probability of applying stretch
        """
        self.stretch_factor = stretch_factor
        self.prob = prob

    def __call__(self, image):
        if random.random() > self.prob:
            return image

        stretch_direction = random.choice(["horizontal", "vertical"])

        if isinstance(image, Image.Image):
            w, h = image.size
            if stretch_direction == "horizontal":
                # Horizontal stretch
                stretched = image.resize((int(w * self.stretch_factor), h))
                # Center crop back to original size
                left = (stretched.size[0] - w) // 2
                cropped = stretched.crop((left, 0, left + w, h))
            else:
                # Vertical stretch
                stretched = image.resize((w, int(h * self.stretch_factor)))
                # Center crop back to original size
                top = (stretched.size[1] - h) // 2
                cropped = stretched.crop((0, top, w, top + h))
            return cropped
        else:
            # Handle numpy arrays
            h, w = image.shape[:2]
            if stretch_direction == "horizontal":
                stretched = cv2.resize(image, (int(w * self.stretch_factor), h))
                start_x = (stretched.shape[1] - w) // 2
                return stretched[:, start_x : start_x + w]
            else:
                stretched = cv2.resize(image, (w, int(h * self.stretch_factor)))
                start_y = (stretched.shape[0] - h) // 2
                return stretched[start_y : start_y + h, :]


class BodyPartCropTransform:
    """
    Applies body part cropping (legs, arms, half-body) similar to augmentation module.
    """

    def __init__(
        self,
        crop_types: List[str] = ["legs", "arms_legs", "half_left", "half_right"],
        prob: float = 0.4,
    ):
        """
        Args:
            crop_types: Types of crops to apply
            prob: Probability of applying crop
        """
        self.crop_types = crop_types
        self.prob = prob

    def __call__(self, image):
        if random.random() > self.prob:
            return image

        crop_type = random.choice(self.crop_types)

        if isinstance(image, Image.Image):
            w, h = image.size

            if crop_type == "legs":
                # Remove bottom 50% (legs)
                cropped = image.crop((0, 0, w, int(h * 0.5)))
            elif crop_type == "arms_legs":
                # Remove arms (15% from sides) and legs (50% from bottom)
                arm_crop = 0.15
                leg_crop = 0.5
                cropped = image.crop(
                    (int(w * arm_crop), 0, int(w * (1 - arm_crop)), int(h * (1 - leg_crop)))
                )
            elif crop_type == "half_left":
                # Left half of body
                cropped = image.crop((0, 0, int(w * 0.5), h))
            elif crop_type == "half_right":
                # Right half of body
                cropped = image.crop((int(w * 0.5), 0, w, h))
            else:
                return image

            return cropped
        else:
            # Handle numpy arrays
            h, w = image.shape[:2]

            if crop_type == "legs":
                return image[: int(h * 0.5), :, :]
            elif crop_type == "arms_legs":
                arm_crop = 0.15
                leg_crop = 0.5
                return image[
                    : int(h * (1 - leg_crop)), int(w * arm_crop) : int(w * (1 - arm_crop)), :
                ]
            elif crop_type == "half_left":
                return image[:, : int(w * 0.5), :]
            elif crop_type == "half_right":
                return image[:, int(w * 0.5) :, :]
            else:
                return image


class RandomOcclusionTransform:
    """
    Applies random occlusion patches to simulate partial object visibility.
    """

    def __init__(
        self,
        num_occlusions: Tuple[int, int] = (3, 10),
        occlusion_ratio: float = 0.25,
        prob: float = 0.3,
    ):
        """
        Args:
            num_occlusions: Range of number of occlusion patches
            occlusion_ratio: Size of occlusion as ratio of min(width, height)
            prob: Probability of applying occlusion
        """
        self.num_occlusions = num_occlusions
        self.occlusion_ratio = occlusion_ratio
        self.prob = prob

    def __call__(self, image):
        if random.random() > self.prob:
            return image

        if isinstance(image, Image.Image):
            image_array = np.array(image)
            occluded = self._apply_occlusion(image_array)
            return Image.fromarray(occluded)
        else:
            return self._apply_occlusion(image.copy())

    def _apply_occlusion(self, image_array):
        h, w = image_array.shape[:2]
        occ_size = max(5, min(15, int(min(h, w) * self.occlusion_ratio)))

        if h < occ_size or w < occ_size:
            return image_array

        num_patches = random.randint(*self.num_occlusions)

        for _ in range(num_patches):
            occ_y = random.randint(0, h - occ_size)
            occ_x = random.randint(0, w - occ_size)
            random_color = [random.randint(0, 255) for _ in range(3)]
            image_array[occ_y : occ_y + occ_size, occ_x : occ_x + occ_size] = random_color

        return image_array


class HSVAdjustmentTransform:
    """
    Applies HSV adjustments for lighting condition variations (darkness/oversaturation).
    """

    def __init__(self, variations: List[str] = ["darken", "oversaturate"], prob: float = 0.4):
        """
        Args:
            variations: Types of HSV adjustments ['darken', 'oversaturate']
            prob: Probability of applying adjustment
        """
        self.variations = variations
        self.prob = prob

    def __call__(self, image):
        if random.random() > self.prob:
            return image

        variation = random.choice(self.variations)

        if isinstance(image, Image.Image):
            image_array = np.array(image)
            adjusted = self._apply_hsv_adjustment(image_array, variation)
            return Image.fromarray(adjusted)
        else:
            return self._apply_hsv_adjustment(image.copy(), variation)

    def _apply_hsv_adjustment(self, image_array, variation):
        # Convert RGB to HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Split channels for safer manipulation
        h, s, v = cv2.split(hsv)
        s = s.astype(np.float32)
        v = v.astype(np.float32)

        if variation == "darken":
            # Reduce saturation and brightness (simulating shadow/darkness)
            s = s * 0.7  # Reduce saturation by 30%
            v = v * 0.6  # Reduce brightness by 40%
        elif variation == "oversaturate":
            # Increase saturation (simulating bright sunlight)
            s = s * 1.3  # Increase saturation by 30%
            v = v * 0.9  # Slightly reduce brightness

        # Clip values to valid range before converting to uint8
        s = np.clip(s, 0, 255).astype(np.uint8)
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        # Merge channels back
        hsv_adjusted = cv2.merge([h, s, v])
        
        # Convert back to RGB
        rgb_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)

        return rgb_adjusted


class MultiAugmentationTransform:
    """
    Composite transform that applies multiple augmentations in sequence.
    This replaces the offline augmentation module functionality.
    """

    def __init__(self, enable_all: bool = True):
        """
        Args:
            enable_all: Whether to enable all augmentation types
        """
        self.transforms = []

        if enable_all:
            # Add all augmentation transforms with probabilities
            self.transforms.extend(
                [
                    RandomRotationTransform(angles=[30, -30], prob=0.3),
                    StretchTransform(stretch_factor=2.0, prob=0.2),
                    BodyPartCropTransform(prob=0.25),
                    RandomOcclusionTransform(prob=0.2),
                    HSVAdjustmentTransform(prob=0.3),
                ]
            )

    def __call__(self, image):
        # Apply transforms with some probability
        for transform in self.transforms:
            image = transform(image)
        return image


def create_training_transforms():
    """Create training transforms with data augmentation."""
    transforms_list = []

    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())

    # Add standard transforms
    transforms_list.extend(
        [
            transforms.Resize((model_config.input_height, model_config.input_width)),
            # transforms.RandomHorizontalFlip(p=transform_config.hflip_prob),
            # transforms.ColorJitter(
            #     brightness=transform_config.colorjitter_brightness,
            #     contrast=transform_config.colorjitter_contrast,
            #     saturation=transform_config.colorjitter_saturation,
            #     hue=transform_config.colorjitter_hue,
            # ),
            # transforms.RandomRotation(transform_config.random_rotation_degrees),
            # transforms.RandomAffine(
            #     degrees=transform_config.random_affine_degrees,
            #     translate=transform_config.random_affine_translate,
            # ),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]
    )

    return transforms.Compose(transforms_list)


def create_advanced_training_transforms():
    """Create advanced training transforms with comprehensive augmentation (replaces offline augmentation)."""
    transforms_list = []

    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())

    # Add advanced augmentation transforms (replaces augmentation module)
    transforms_list.extend(
        [
            # Multi-augmentation transform that combines various techniques
            MultiAugmentationTransform(enable_all=True),
            # Resize to target size
            transforms.Resize((model_config.input_height, model_config.input_width)),
            # Standard transforms
            transforms.RandomHorizontalFlip(p=transform_config.hflip_prob),
            transforms.ColorJitter(
                brightness=transform_config.colorjitter_brightness,
                contrast=transform_config.colorjitter_contrast,
                saturation=transform_config.colorjitter_saturation,
                hue=transform_config.colorjitter_hue,
            ),
            transforms.RandomRotation(transform_config.random_rotation_degrees),
            transforms.RandomAffine(
                degrees=transform_config.random_affine_degrees,
                translate=transform_config.random_affine_translate,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]
    )

    return transforms.Compose(transforms_list)


def create_inference_transforms():
    """Create inference transforms without augmentation."""
    transforms_list = []

    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())

    # Add standard transforms
    transforms_list.extend(
        [
            transforms.Resize((model_config.input_height, model_config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]
    )

    return transforms.Compose(transforms_list)


def create_tensor_to_pil_transforms():
    """Create transforms for converting tensor back to PIL Image."""
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / s for s in model_config.imagenet_std]
            ),
            transforms.Normalize(
                mean=[-m for m in model_config.imagenet_mean], std=[1.0, 1.0, 1.0]
            ),
            transforms.ToPILImage(),
        ]
    )


def create_opencv_safe_transforms():
    """Create transforms that handle OpenCV BGR input safely."""
    transforms_list = [
        transforms.Lambda(
            lambda x: ensure_rgb_format(x, "BGR") if isinstance(x, np.ndarray) else x
        ),
        transforms.ToPILImage(),
    ]

    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())

    # Add standard transforms
    transforms_list.extend(
        [
            transforms.Resize((model_config.input_height, model_config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]
    )

    return transforms.Compose(transforms_list)


def create_opencv_safe_training_transforms():
    """Create training transforms that handle OpenCV BGR input safely."""
    transforms_list = [
        transforms.Lambda(
            lambda x: ensure_rgb_format(x, "BGR") if isinstance(x, np.ndarray) else x
        ),
        transforms.ToPILImage(),
    ]

    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())

    # Add standard transforms
    transforms_list.extend(
        [
            transforms.Resize((model_config.input_height, model_config.input_width)),
            transforms.RandomHorizontalFlip(p=transform_config.hflip_prob),
            transforms.ColorJitter(
                brightness=transform_config.colorjitter_brightness,
                contrast=transform_config.colorjitter_contrast,
                saturation=transform_config.colorjitter_saturation,
                hue=transform_config.colorjitter_hue,
            ),
            transforms.RandomRotation(transform_config.random_rotation_degrees),
            transforms.RandomAffine(
                degrees=transform_config.random_affine_degrees,
                translate=transform_config.random_affine_translate,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]
    )

    return transforms.Compose(transforms_list)


def create_opencv_safe_advanced_training_transforms():
    """Create advanced training transforms that handle OpenCV BGR input safely (replaces offline augmentation)."""
    transforms_list = [
        transforms.Lambda(
            lambda x: ensure_rgb_format(x, "BGR") if isinstance(x, np.ndarray) else x
        ),
        transforms.ToPILImage(),
    ]

    # Add background removal if enabled
    if transform_config.enable_background_removal:
        transforms_list.append(BackgroundRemovalTransform())

    # Add advanced augmentation transforms (replaces augmentation module)
    transforms_list.extend(
        [
            # Multi-augmentation transform that combines various techniques
            MultiAugmentationTransform(enable_all=True),
            # Resize to target size
            transforms.Resize((model_config.input_height, model_config.input_width)),
            # Standard transforms
            transforms.RandomHorizontalFlip(p=transform_config.hflip_prob),
            transforms.ColorJitter(
                brightness=transform_config.colorjitter_brightness,
                contrast=transform_config.colorjitter_contrast,
                saturation=transform_config.colorjitter_saturation,
                hue=transform_config.colorjitter_hue,
            ),
            transforms.RandomRotation(transform_config.random_rotation_degrees),
            transforms.RandomAffine(
                degrees=transform_config.random_affine_degrees,
                translate=transform_config.random_affine_translate,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]
    )

    return transforms.Compose(transforms_list)


def refresh_transform_instances():
    """
    Refresh all transform instances to pick up configuration changes.
    Call this after modifying transform configuration or background detector.
    """
    global training_transforms, inference_transforms, validation_transforms
    global opencv_safe_transforms, opencv_safe_training_transforms
    global advanced_training_transforms, opencv_safe_advanced_training_transforms

    # Recreate all transform instances
    training_transforms = create_training_transforms()
    advanced_training_transforms = create_advanced_training_transforms()
    inference_transforms = create_inference_transforms()
    validation_transforms = inference_transforms  # Same as inference
    opencv_safe_transforms = create_opencv_safe_transforms()
    opencv_safe_training_transforms = create_opencv_safe_training_transforms()
    opencv_safe_advanced_training_transforms = create_opencv_safe_advanced_training_transforms()

    # Update the transforms dictionary
    TRANSFORMS.update(
        {
            "training": training_transforms,
            "advanced_training": advanced_training_transforms,
            "inference": inference_transforms,
            "validation": validation_transforms,
            "opencv_safe": opencv_safe_transforms,
            "opencv_safe_training": opencv_safe_training_transforms,
            "opencv_safe_advanced_training": opencv_safe_advanced_training_transforms,
        }
    )


# Create transform instances
training_transforms = create_training_transforms()
advanced_training_transforms = create_advanced_training_transforms()
inference_transforms = create_inference_transforms()
validation_transforms = inference_transforms  # Same as inference
tensor_to_pil = create_tensor_to_pil_transforms()
opencv_safe_transforms = create_opencv_safe_transforms()
opencv_safe_training_transforms = create_opencv_safe_training_transforms()
opencv_safe_advanced_training_transforms = create_opencv_safe_advanced_training_transforms()

# Dictionary for easy access to all transforms
TRANSFORMS = {
    "training": training_transforms,
    "advanced_training": advanced_training_transforms,
    "inference": inference_transforms,
    "validation": validation_transforms,
    "tensor_to_pil": tensor_to_pil,
    "opencv_safe": opencv_safe_transforms,
    "opencv_safe_training": opencv_safe_training_transforms,
    "opencv_safe_advanced_training": opencv_safe_advanced_training_transforms,
}


def create_transforms_with_background_removal(background_detector: "BackgroundMaskDetector"):
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
            "training": create_training_transforms(),
            "inference": create_inference_transforms(),
            "validation": create_inference_transforms(),
            "tensor_to_pil": tensor_to_pil,  # No background removal needed for tensor->PIL
            "opencv_safe": create_opencv_safe_transforms(),
            "opencv_safe_training": create_opencv_safe_training_transforms(),
        }

        return result

    finally:
        # Restore original settings
        transform_config.enable_background_removal = old_enabled
        set_global_background_detector(old_detector)


def get_transforms(
    mode="inference", background_detector: Optional["BackgroundMaskDetector"] = None
):
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
            raise ValueError(
                f"Unknown transform mode: {mode}. Available modes: {list(bg_transforms.keys())}"
            )
        return bg_transforms[mode]
    else:
        # Use the new centralized approach
        if mode not in TRANSFORMS:
            raise ValueError(
                f"Unknown transform mode: {mode}. Available modes: {list(TRANSFORMS.keys())}"
            )
        return TRANSFORMS[mode]
