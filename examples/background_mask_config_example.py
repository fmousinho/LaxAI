"""
Example usage of BackgroundMaskDetector with configuration.

This example demonstrates how to use the BackgroundMaskDetector with different
configuration options.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from modules.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
from config.all_config import BackgroundMaskConfig, background_mask_config

def main():
    print("BackgroundMaskDetector Configuration Examples")
    print("=" * 50)
    
    # Create test frames with green background
    test_frames = []
    for i in range(8):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = [0, 180, 0]  # Green background
        
        # Add some objects
        frame[40:80, 40:80] = [255, 0, 0]  # Red square
        frame[20:40, 100:140] = [0, 0, 255]  # Blue rectangle
        
        test_frames.append(frame)
    
    # Example 1: Using global default configuration
    print("\n1. Using Global Default Configuration:")
    print(f"   Sample frames: {background_mask_config.sample_frames}")
    print(f"   Std dev multiplier: {background_mask_config.std_dev_multiplier}")
    print(f"   Replacement color: {background_mask_config.replacement_color}")
    
    frame_gen1 = create_frame_generator_from_images(test_frames)
    detector1 = BackgroundMaskDetector(frame_generator=frame_gen1, verbose=False)
    result1 = detector1.remove_background(test_frames[0])
    print(f"   ✓ Background removal completed")
    
    # Example 2: Using custom configuration
    print("\n2. Using Custom Configuration:")
    custom_config = BackgroundMaskConfig(
        sample_frames=3,
        std_dev_multiplier=1.5,
        replacement_color=(128, 128, 128),  # Gray
        top_crop_ratio=0.4,  # Remove top 40%
        bottom_crop_ratio=0.15,  # Remove bottom 15%
        verbose=False
    )
    
    print(f"   Sample frames: {custom_config.sample_frames}")
    print(f"   Std dev multiplier: {custom_config.std_dev_multiplier}")
    print(f"   Replacement color: {custom_config.replacement_color}")
    print(f"   Top crop ratio: {custom_config.top_crop_ratio}")
    print(f"   Bottom crop ratio: {custom_config.bottom_crop_ratio}")
    
    frame_gen2 = create_frame_generator_from_images(test_frames)
    detector2 = BackgroundMaskDetector(frame_generator=frame_gen2, config=custom_config)
    result2 = detector2.remove_background(test_frames[0])
    print(f"   ✓ Background removal completed with custom config")
    
    # Example 3: Parameter override
    print("\n3. Using Parameter Override:")
    frame_gen3 = create_frame_generator_from_images(test_frames)
    detector3 = BackgroundMaskDetector(
        frame_generator=frame_gen3,
        sample_frames=2,  # Override config
        replacement_color=(255, 255, 0),  # Yellow
        verbose=False
    )
    
    result3 = detector3.remove_background(test_frames[0])
    print(f"   Sample frames (override): 2")
    print(f"   Replacement color (override): (255, 255, 0)")
    print(f"   ✓ Background removal completed with parameter override")
    
    # Example 4: Dynamic configuration updates
    print("\n4. Dynamic Configuration Updates:")
    frame_gen4 = create_frame_generator_from_images(test_frames)
    detector4 = BackgroundMaskDetector(frame_generator=frame_gen4, verbose=False)
    
    # Change replacement color
    detector4.update_replacement_color((255, 0, 255))  # Magenta
    print(f"   ✓ Replacement color updated to magenta")
    
    # Change bounds sensitivity
    detector4.update_bounds(0.5)  # More strict
    print(f"   ✓ Bounds updated with std multiplier: 0.5")
    
    result4 = detector4.remove_background(test_frames[0])
    print(f"   ✓ Background removal completed with updated settings")
    
    # Example 5: Get statistics
    print("\n5. Configuration Statistics:")
    stats = detector4.get_stats()
    print(f"   Mean HSV: {stats['mean_hsv']}")
    print(f"   Std HSV: {stats['std_hsv']}")
    print(f"   Current std multiplier: {stats['std_dev_multiplier']}")
    print(f"   Current replacement color: {stats['replacement_color']}")
    print(f"   Config limits: {stats['config']['min_std_multiplier']:.1f} - {stats['config']['max_std_multiplier']:.1f}")
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")

if __name__ == "__main__":
    main()
