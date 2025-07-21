#!/usr/bin/env python3
"""
Test the new background removal folder structure.
"""

import os
import tempfile
import numpy as np
import cv2

def test_crop_path_parsing():
    """Test parsing crop paths and creating new folder structure."""
    
    # Simulate the rel_path format from the original crops
    test_rel_paths = [
        "frame_123/123_456_0.850.jpg",
        "frame_78/78_789_0.923.jpg", 
        "frame_0/0_111_0.750.jpg"
    ]
    
    print("=== Testing Crop Path Parsing ===")
    
    for rel_path in test_rel_paths:
        print(f"\nOriginal rel_path: {rel_path}")
        
        # Parse crop information from rel_path (format: frame_{frame_id}/{frame_id}_{tracker_id}_{confidence}.jpg)
        path_parts = rel_path.split('/')
        if len(path_parts) >= 2:
            frame_folder = path_parts[0]  # e.g., "frame_123"
            crop_filename = path_parts[1]  # e.g., "123_456_0.850.jpg"
            
            # Extract frame_id and tracker_id from filename
            name_parts = crop_filename.split('_')
            if len(name_parts) >= 2:
                frame_id = name_parts[0]
                tracker_id = name_parts[1]
                
                # Create new path structure: video_folder/crops/modified/frame{frame_id}/crop_{tracker_id}/crop.jpg
                modified_crops_folder = "video_12345/crops/modified"
                new_storage_path = f"{modified_crops_folder}/frame{frame_id}/crop_{tracker_id}/crop.jpg"
                
                print(f"  Frame ID: {frame_id}")
                print(f"  Tracker ID: {tracker_id}")
                print(f"  New storage path: {new_storage_path}")
            else:
                print(f"  ERROR: Invalid crop filename format: {crop_filename}")
        else:
            print(f"  ERROR: Invalid crop path format: {rel_path}")

def create_test_crop_structure():
    """Create a test crop structure and simulate background removal."""
    
    print("\n=== Creating Test Crop Structure ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary directory: {temp_dir}")
        
        # Simulate the original crops structure
        original_crops = {
            "frame_10/10_100_0.950.jpg": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "frame_10/10_200_0.875.jpg": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "frame_25/25_100_0.890.jpg": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        }
        
        # Create original crops directory structure
        original_dir = os.path.join(temp_dir, "original")
        for rel_path, crop_img in original_crops.items():
            full_path = os.path.join(original_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Convert RGB to BGR for saving
            crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(full_path, crop_bgr)
            print(f"Created original crop: {rel_path}")
        
        # Simulate processing crops with new folder structure
        modified_dir = os.path.join(temp_dir, "modified")
        
        for rel_path, crop_img in original_crops.items():
            # Parse crop information
            path_parts = rel_path.split('/')
            if len(path_parts) >= 2:
                frame_folder = path_parts[0]  # e.g., "frame_10"
                crop_filename = path_parts[1]  # e.g., "10_100_0.950.jpg"
                
                # Extract frame_id and tracker_id from filename
                name_parts = crop_filename.split('_')
                if len(name_parts) >= 2:
                    frame_id = name_parts[0]
                    tracker_id = name_parts[1]
                    
                    # Create new path structure
                    new_rel_path = f"frame{frame_id}/crop_{tracker_id}/crop.jpg"
                    new_full_path = os.path.join(modified_dir, new_rel_path)
                    
                    # Create directory and save
                    os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
                    
                    # Simulate background removal (just add some white pixels for demo)
                    processed_crop = crop_img.copy()
                    processed_crop[0:10, 0:10] = [255, 255, 255]  # Add white corner
                    
                    # Save processed crop
                    crop_bgr = cv2.cvtColor(processed_crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(new_full_path, crop_bgr)
                    print(f"Created modified crop: {new_rel_path}")
        
        # Verify the structure
        print(f"\nVerifying structure in {temp_dir}:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

if __name__ == "__main__":
    test_crop_path_parsing()
    create_test_crop_structure()
