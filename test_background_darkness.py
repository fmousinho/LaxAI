#!/usr/bin/env python3
"""
Test to debug background removal darkness issue.
"""

import numpy as np
import cv2
from core.common.background_mask import BackgroundMaskDetector, create_frame_generator_from_images

def create_test_images():
    """Create test images with known colors."""
    # Create a test image: green background with red and blue objects
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Green background (grass-like)
    img[:, :] = [34, 139, 34]  # Forest green in RGB
    
    # Red player (top half)
    img[20:40, 30:70] = [255, 0, 0]  # Red in RGB
    
    # Blue player (bottom half)  
    img[60:80, 30:70] = [0, 0, 255]  # Blue in RGB
    
    return img

def test_background_removal_formats():
    """Test background removal with different input formats."""
    print("=== Testing Background Removal Darkness Issue ===")
    
    # Create test image in RGB
    test_img_rgb = create_test_images()
    print(f"Original RGB image stats:")
    print(f"  Shape: {test_img_rgb.shape}")
    print(f"  Min: {test_img_rgb.min()}, Max: {test_img_rgb.max()}")
    print(f"  Background pixel sample (RGB): {test_img_rgb[10, 10]}")
    print(f"  Red player pixel sample (RGB): {test_img_rgb[30, 50]}")
    print(f"  Blue player pixel sample (RGB): {test_img_rgb[70, 50]}")
    
    # Convert to BGR for testing
    test_img_bgr = cv2.cvtColor(test_img_rgb, cv2.COLOR_RGB2BGR)
    print(f"\nBGR version stats:")
    print(f"  Background pixel sample (BGR): {test_img_bgr[10, 10]}")
    print(f"  Red player pixel sample (BGR): {test_img_bgr[30, 50]}")
    print(f"  Blue player pixel sample (BGR): {test_img_bgr[70, 50]}")
    
    # Create frame generator with multiple similar images
    test_frames = [test_img_rgb] * 5
    frame_gen = create_frame_generator_from_images(test_frames, input_format='RGB')
    
    # Initialize detector
    detector = BackgroundMaskDetector(verbose=True, sample_frames=3)
    detector.initialize(frame_gen)
    
    print(f"\nDetector stats:")
    stats = detector.get_stats()
    print(f"  Mean HSV: {stats['mean_hsv']}")
    print(f"  Std HSV: {stats['std_hsv']}")
    print(f"  Lower bound: {stats['lower_bound']}")
    print(f"  Upper bound: {stats['upper_bound']}")
    print(f"  Replacement color (RGB): {stats['replacement_color']}")
    
    # Test with RGB input
    print(f"\n=== Testing RGB Input ===")
    result_rgb = detector.remove_background(test_img_rgb, input_format='RGB')
    print(f"Result RGB stats:")
    print(f"  Shape: {result_rgb.shape}")
    print(f"  Min: {result_rgb.min()}, Max: {result_rgb.max()}")
    print(f"  Background pixel sample: {result_rgb[10, 10]}")
    print(f"  Red player pixel sample: {result_rgb[30, 50]}")
    print(f"  Blue player pixel sample: {result_rgb[70, 50]}")
    
    # Test with BGR input
    print(f"\n=== Testing BGR Input ===")
    result_bgr = detector.remove_background(test_img_bgr, input_format='BGR')
    print(f"Result BGR stats:")
    print(f"  Shape: {result_bgr.shape}")
    print(f"  Min: {result_bgr.min()}, Max: {result_bgr.max()}")
    print(f"  Background pixel sample: {result_bgr[10, 10]}")
    print(f"  Red player pixel sample: {result_bgr[30, 50]}")
    print(f"  Blue player pixel sample: {result_bgr[70, 50]}")
    
    # Check the mask directly
    print(f"\n=== Checking Mask ===")
    mask_rgb = detector.get_background_mask(test_img_rgb, input_format='RGB')
    mask_bgr = detector.get_background_mask(test_img_bgr, input_format='BGR')
    
    background_pixels_rgb = np.sum(mask_rgb > 0)
    background_pixels_bgr = np.sum(mask_bgr > 0)
    total_pixels = mask_rgb.size
    
    print(f"RGB mask: {background_pixels_rgb}/{total_pixels} background pixels ({100*background_pixels_rgb/total_pixels:.1f}%)")
    print(f"BGR mask: {background_pixels_bgr}/{total_pixels} background pixels ({100*background_pixels_bgr/total_pixels:.1f}%)")
    
    # Check specific pixels in mask
    print(f"Mask values at key locations:")
    print(f"  Background (10,10): RGB={mask_rgb[10,10]}, BGR={mask_bgr[10,10]}")
    print(f"  Red player (30,50): RGB={mask_rgb[30,50]}, BGR={mask_bgr[30,50]}")
    print(f"  Blue player (70,50): RGB={mask_rgb[70,50]}, BGR={mask_bgr[70,50]}")

if __name__ == "__main__":
    test_background_removal_formats()
