#!/usr/bin/env python3
"""
Test script to verify RGB background removal functionality
"""

import sys
import os
sys.path.append('.')

import numpy as np
import cv2
import tempfile
from core.common.background_mask import BackgroundMaskDetector, create_frame_generator_from_images

def test_rgb_background_removal():
    """Test RGB background removal with crop-like images"""
    
    print("=== Testing RGB Background Removal ===")
    
    # Create detector and initialize with grass-like frames
    detector = BackgroundMaskDetector()
    
    # Create realistic grass frames (BGR format for initialization)
    grass_frames = []
    for i in range(3):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 1] = 120  # Green channel in BGR
        grass_frames.append(frame)
    
    frame_generator = create_frame_generator_from_images(grass_frames, input_format='BGR')
    detector.initialize(frame_generator)
    
    print("✓ Background detector initialized")
    
    # Create test RGB crops (as they would be stored after extraction)
    test_crops = []
    
    # Crop 1: Green background with red player
    crop1 = np.zeros((100, 100, 3), dtype=np.uint8)
    crop1[:, :, 1] = 120  # Green background in RGB
    crop1[30:70, 30:70, 0] = 255  # Red player in RGB
    crop1[30:70, 30:70, 1] = 0    # Remove green from player
    test_crops.append(("green_bg_red_player", crop1))
    
    # Crop 2: Green background with blue player
    crop2 = np.zeros((100, 100, 3), dtype=np.uint8)
    crop2[:, :, 1] = 120  # Green background in RGB
    crop2[30:70, 30:70, 2] = 255  # Blue player in RGB
    crop2[30:70, 30:70, 1] = 0    # Remove green from player
    test_crops.append(("green_bg_blue_player", crop2))
    
    # Create temp directory for results
    temp_dir = tempfile.mkdtemp()
    print(f"Saving test results to: {temp_dir}")
    
    for crop_name, crop_rgb in test_crops:
        print(f"\n--- Testing {crop_name} ---")
        
        # Save original crop
        original_path = os.path.join(temp_dir, f"{crop_name}_original.jpg")
        crop_bgr_for_save = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_path, crop_bgr_for_save)
        
        print(f"Original crop pixel [0,0] (RGB): {crop_rgb[0,0]}")
        print(f"Original crop pixel [50,50] (RGB): {crop_rgb[50,50]}")
        
        # Test background removal
        try:
            processed_rgb = detector.remove_background(crop_rgb, input_format='RGB')
            
            print(f"Processed crop pixel [0,0] (RGB): {processed_rgb[0,0]}")
            print(f"Processed crop pixel [50,50] (RGB): {processed_rgb[50,50]}")
            
            # Save processed crop (RGB→BGR conversion)
            processed_path = os.path.join(temp_dir, f"{crop_name}_processed.jpg")
            processed_bgr_for_save = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(processed_path, processed_bgr_for_save)
            
            # Verify by reading back and converting to RGB
            loaded_bgr = cv2.imread(processed_path)
            loaded_rgb = cv2.cvtColor(loaded_bgr, cv2.COLOR_BGR2RGB)
            
            print(f"Loaded back pixel [0,0] (RGB): {loaded_rgb[0,0]}")
            print(f"Loaded back pixel [50,50] (RGB): {loaded_rgb[50,50]}")
            
            # Check if background was removed (should be white)
            background_removed = np.all(loaded_rgb[0,0] > 200)
            player_preserved = loaded_rgb[50,50,0] > 100 or loaded_rgb[50,50,2] > 100  # Red or blue
            
            print(f"✓ Background removed: {background_removed}")
            print(f"✓ Player preserved: {player_preserved}")
            
            if background_removed and player_preserved:
                print(f"✓ {crop_name} test PASSED")
            else:
                print(f"✗ {crop_name} test FAILED")
                
        except Exception as e:
            print(f"✗ Error processing {crop_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTest results saved in: {temp_dir}")
    print("=== RGB Background Removal Test Complete ===")

if __name__ == "__main__":
    test_rgb_background_removal()
