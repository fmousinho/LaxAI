"""
Image augmentation utilities for the LaxAI project.

This module provides functions for performing various augmentations on RGB images,
particularly useful for training embeddings models with enhanced data diversity.
"""

import cv2
import numpy as np
import random
from typing import List


def augment_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Perform various augmentations on a list of RGB images using OpenCV.
    
    This function applies multiple augmentation techniques to increase data diversity:
    - Horizontal flips
    - Rotations (+/- 20 degrees)
    - Stretching (horizontal and vertical)
    - Random crops with different sizes
    - Random occlusion
    - Gaussian noise
    - HSV adjustments (darkness/oversaturation)
    
    Args:
        images: List of RGB numpy arrays (images)
        
    Returns:
        List of augmented RGB numpy arrays
    """
    augmented_images = []
    
    for img in images:
        # Skip invalid or very small images
        if img is None or img.size == 0 or len(img.shape) < 3:
            continue
            
        h, w = img.shape[:2]

        
        # Original image
        augmented_images.append(img.copy())
        
        # # 1. Left-to-right flip
        # flipped = cv2.flip(img, 1)
        # augmented_images.append(flipped)

        # # 2. Rotations (+30 and -30 degrees)
        # center = (w // 2, h // 2)

        # # +30 degree rotation
        # rotation_matrix_pos = cv2.getRotationMatrix2D(center, 30, 1.0)
        # rotated_pos = cv2.warpAffine(img, rotation_matrix_pos, (w, h), borderMode=cv2.BORDER_REFLECT)
        # augmented_images.append(rotated_pos)

        # # -30 degree rotation
        # rotation_matrix_neg = cv2.getRotationMatrix2D(center, -30, 1.0)
        # rotated_neg = cv2.warpAffine(img, rotation_matrix_neg, (w, h), borderMode=cv2.BORDER_REFLECT)
        # augmented_images.append(rotated_neg)
        
        # # # 3. 100% stretching (horizontal and vertical)
        # stretch_factor = 2.0
        
        # # Horizontal stretch
        # stretched_h = cv2.resize(img, (int(w * stretch_factor), h))
        # # Crop back to original size from center
        # start_x = (stretched_h.shape[1] - w) // 2
        # stretched_h_cropped = stretched_h[:, start_x:start_x + w]
        # augmented_images.append(stretched_h_cropped)
        
        # # Vertical stretch
        # stretched_v = cv2.resize(img, (w, int(h * stretch_factor)))
        # # Crop back to original size from center
        # start_y = (stretched_v.shape[0] - h) // 2
        # stretched_v_cropped = stretched_v[start_y:start_y + h, :]
        # augmented_images.append(stretched_v_cropped)
        
        # 4. Crops
        
        #4.1 Remove legs

        LEG_CROP_VALUES = [.5, .4, .3]
        for LEG_CROP in LEG_CROP_VALUES:
            leg_cropped = img[:int(h - h * LEG_CROP), :, :]
            augmented_images.append(leg_cropped.copy())

        #4.2 Leg and arms cropped
        ARM_CROP = .15
        arms_cropped = img[:int(h - h * LEG_CROP), int(w * ARM_CROP):int(w * (1 - ARM_CROP)), :]  # Crop arms (top 15% pixels) and legs
        augmented_images.append(arms_cropped.copy())

        #4.3 Half body crop
        HALF_CROP = .5
        half_cropped = img[:, :int(w * (1 - HALF_CROP)), :]  # Left side of body
        augmented_images.append(half_cropped.copy())
        half_cropped_lr = img[:, int(w * HALF_CROP):, :]  # Right side of body
        augmented_images.append(half_cropped_lr.copy())

        #4.4 Upper body crop
        UPPER_CROP = .3
        upper_cropped = img[:int(h * (1 - UPPER_CROP)), :, :]  # Upper body
        augmented_images.append(upper_cropped.copy())

 
        # 5. Random occlusion - adaptive size based on image dimensions
        OCC_SAMPLES = 5  # Number of occluded images generated

        # Calculate occlusion size as percentage of image (minimum 5x5, maximum 15x15)
        occ_size = min(15, max(5, min(h, w) // 4))
        
        # Only add occlusion if image is large enough
        if h >= occ_size and w >= occ_size:
            for _ in range(OCC_SAMPLES):
                occluded = img.copy()
                num_occlusions = random.randint(3, 10)  # 3-10 occlusions per image
                for _ in range(num_occlusions):
                    occ_y = random.randint(0, h - occ_size)
                    occ_x = random.randint(0, w - occ_size)
                    # Fill with random color
                    random_color = [random.randint(0, 255) for _ in range(3)]
                    occluded[occ_y:occ_y + occ_size, occ_x:occ_x + occ_size] = random_color
                augmented_images.append(occluded.copy())
        
        # 6. Noise (Gaussian noise)
        # noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        # noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # augmented_images.append(noisy)
        
        # # 7. Saturation/Darkness (simulating sun exposure)
        # # Convert RGB to HSV for saturation adjustment
        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # hsv_float = hsv.astype(np.float32)
        
        # # Reduce saturation and value (brightness)
        # hsv_float[:, :, 1] *= 0.7  # Reduce saturation by 30%
        # hsv_float[:, :, 2] *= 0.6  # Reduce brightness by 40%
        
        # # Clip values and convert back
        # hsv_adjusted = np.clip(hsv_float, 0, 255).astype(np.uint8)
        # darkened = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
        # augmented_images.append(darkened)
        
        # # 8. Additional saturation variation (oversaturated)
        # hsv_over = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # hsv_over_float = hsv_over.astype(np.float32)
        
        # # Increase saturation
        # hsv_over_float[:, :, 1] *= 1.3  # Increase saturation by 30%
        # hsv_over_float[:, :, 2] *= 0.9  # Slightly reduce brightness
        
        # hsv_over_adjusted = np.clip(hsv_over_float, 0, 255).astype(np.uint8)
        # oversaturated = cv2.cvtColor(hsv_over_adjusted, cv2.COLOR_HSV2RGB)
        # augmented_images.append(oversaturated)
    
    return augmented_images


def test_augmentation() -> List[np.ndarray]:
    """
    Test the augmentation function with a sample image.
    
    Creates a simple test image (red square) and applies augmentations to verify
    the function works correctly.
    
    Returns:
        List of augmented images from the test
    """
    # Create a simple test image (red square)
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[25:75, 25:75] = [255, 0, 0]  # Red square in RGB
    
    # Test with single image
    augmented = augment_images([test_img])
    print(f"Test augmentation: 1 image -> {len(augmented)} augmented images")
    
    return augmented





if __name__ == "__main__":
    # Run test when module is executed directly
    test_results = test_augmentation()
    print(f"Augmentation module test completed. Generated {len(test_results)} images.")
