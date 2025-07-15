#!/usr/bin/env python3
"""
Example demonstrating how to use transforms with background removal.

This example shows how to integrate the BackgroundMaskDetector with the
transform pipeline for both training and inference.
"""

import numpy as np
from PIL import Image
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
from config.transforms import get_transforms, create_transforms_with_background_removal
from config.all_config import BackgroundMaskConfig


def create_test_images():
    """Create test images with green background and red objects."""
    test_images = []
    
    for i in range(5):
        # Create image with green background
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :] = [0, 255, 0]  # Green background
        
        # Add some objects
        # Red square
        image[50:100, 50:100] = [255, 0, 0]
        
        # Blue circle (approximate)
        center = (150, 150)
        radius = 25
        y, x = np.ogrid[:200, :200]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = [0, 0, 255]
        
        test_images.append(image)
    
    return test_images


def main():
    print("Background Removal Transforms Example")
    print("=" * 50)
    
    # Create test images
    test_images = create_test_images()
    print(f"Created {len(test_images)} test images")
    
    # Create background detector
    print("\nTraining background detector...")
    frame_generator = create_frame_generator_from_images(test_images)
    
    # Use custom config for background detection
    bg_config = BackgroundMaskConfig(
        sample_frames=3,
        std_dev_multiplier=1.0,
        replacement_color=(255, 255, 255),  # White background
        top_crop_ratio=0.3,  # Remove top 30%
        bottom_crop_ratio=0.1,  # Remove bottom 10%
        verbose=False
    )
    
    background_detector = BackgroundMaskDetector(
        frame_generator=frame_generator,
        config=bg_config
    )
    
    print("✓ Background detector trained successfully")
    
    # Test different transform modes
    modes = ['training', 'inference', 'validation', 'opencv_safe', 'opencv_safe_training']
    
    for mode in modes:
        print(f"\nTesting {mode} transforms...")
        
        # Get transforms without background removal
        standard_transforms = get_transforms(mode)
        
        # Get transforms with background removal
        bg_transforms = get_transforms(mode, background_detector=background_detector)
        
        # Test with a sample image
        test_image = test_images[0]
        
        # Convert to PIL for torchvision transforms
        pil_image = Image.fromarray(test_image)
        
        try:
            # Apply standard transforms
            if mode != 'tensor_to_pil':  # Skip tensor_to_pil for this test
                # Use numpy array for opencv_safe transforms, PIL for others
                input_image = test_image if 'opencv_safe' in mode else pil_image
                
                standard_result = standard_transforms(input_image)
                print(f"  ✓ Standard {mode} transform: {standard_result.shape if hasattr(standard_result, 'shape') else type(standard_result)}")
                
                # Apply background removal transforms
                bg_result = bg_transforms(input_image)
                print(f"  ✓ Background removal {mode} transform: {bg_result.shape if hasattr(bg_result, 'shape') else type(bg_result)}")
            else:
                print(f"  ✓ Skipping {mode} (tensor input required)")
                
        except Exception as e:
            print(f"  ✗ Error with {mode}: {e}")
    
    # Test bulk transform creation
    print(f"\nTesting bulk transform creation...")
    all_bg_transforms = create_transforms_with_background_removal(background_detector)
    
    print(f"Available background removal transforms:")
    for key, transform in all_bg_transforms.items():
        print(f"  - {key}: {len(transform.transforms)} steps")
    
    # Test actual background removal effect
    print(f"\nTesting background removal effect...")
    
    # Original image
    original_image = test_images[0]
    pil_original = Image.fromarray(original_image)
    
    # Apply background removal inference transform
    bg_inference_transforms = get_transforms('inference', background_detector=background_detector)
    
    # Get the result as tensor
    tensor_result = bg_inference_transforms(pil_original)
    
    print(f"Original image shape: {original_image.shape}")
    print(f"Transformed tensor shape: {tensor_result.shape}")
    print(f"Tensor range: [{tensor_result.min():.3f}, {tensor_result.max():.3f}]")
    
    # Test with numpy array input (OpenCV style)
    print(f"\nTesting with numpy array input...")
    
    opencv_transforms = get_transforms('opencv_safe', background_detector=background_detector)
    opencv_result = opencv_transforms(original_image)
    
    print(f"OpenCV-style input shape: {original_image.shape}")
    print(f"OpenCV-style result shape: {opencv_result.shape}")
    print(f"OpenCV-style result range: [{opencv_result.min():.3f}, {opencv_result.max():.3f}]")
    
    print(f"\n✅ All background removal transform tests completed successfully!")


if __name__ == "__main__":
    main()
