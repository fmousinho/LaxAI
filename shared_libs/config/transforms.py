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

from .all_config import (model_config, transform_config)



def create_training_transforms():
    """Create training transforms with data augmentation."""

    # Add standard transforms
    transforms_list = [
            transforms.Resize((model_config.input_height, model_config.input_width)),
            transforms.RandomResizedCrop(
                size=(model_config.input_height, model_config.input_width),
                scale=(0.0, .2),
            ),
            transforms.ColorJitter(
                brightness=transform_config.colorjitter_brightness,
                contrast=transform_config.colorjitter_contrast,
                saturation=transform_config.colorjitter_saturation,
                hue=transform_config.colorjitter_hue,
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)
            ),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]


    return transforms.Compose(transforms_list)


def create_inference_transforms():
    transforms_list = [
            transforms.Resize((model_config.input_height, model_config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]

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


def inference_for_non_pil_transforms():

    transforms_list = [
            transforms.ToPILImage(),
            transforms.Resize((model_config.input_height, model_config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.imagenet_mean, std=model_config.imagenet_std),
        ]
    return transforms.Compose(transforms_list)


# Dictionary for easy access to all transforms
TRANSFORMS = {
    "training": create_training_transforms(),
    "inference": create_inference_transforms(),
    "validation": create_inference_transforms(),
    "tensor_to_pil": create_tensor_to_pil_transforms(),
    "inference_for_non_pil": inference_for_non_pil_transforms(),
}


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
