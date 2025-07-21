#!/usr/bin/env python3
"""
Test the new augmentation folder structure implementation.
"""

import os
import tempfile
import numpy as np
import cv2
from core.train.augmentation import augment_images

def test_augmentation_folder_structure():
    """Test the augmentation and folder structure logic."""
    
    print("=== Testing Augmentation Folder Structure ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary directory: {temp_dir}")
        
        # Create a sample crop image
        test_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Simulate the rel_path from background removal step
        rel_path = "frame_123/123_456_0.850.jpg"
        
        # Parse crop information from rel_path
        path_parts = rel_path.split('/')
        if len(path_parts) >= 2:
            frame_folder = path_parts[0]  # e.g., "frame_123"
            crop_filename = path_parts[1]  # e.g., "123_456_0.850.jpg"
            
            # Extract frame_id and tracker_id from filename
            name_parts = crop_filename.split('_')
            if len(name_parts) >= 2:
                frame_id = name_parts[0]
                tracker_id = name_parts[1]
                
                print(f"Parsed: frame_id={frame_id}, tracker_id={tracker_id}")
                
                # Test augmentation
                augmented_images = augment_images([test_crop])
                print(f"Generated {len(augmented_images)} augmented images (including original)")
                
                # Simulate the new folder structure
                modified_crops_folder = f"{temp_dir}/crops/modified"
                
                # Create folders and save images as the pipeline would
                for i, aug_img in enumerate(augmented_images):
                    if i == 0:
                        # Original crop (already processed with background removal)
                        storage_path = f"{modified_crops_folder}/frame{frame_id}/crop_{tracker_id}/crop.jpg"
                    else:
                        # Augmented crops
                        storage_path = f"{modified_crops_folder}/frame{frame_id}/crop_{tracker_id}/crop_aug{i}.jpg"
                    
                    # Create directory structure
                    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
                    
                    # Save the image
                    aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(storage_path, aug_bgr)
                    print(f"Saved: {storage_path}")
                
                # Verify the structure
                print(f"\nFinal directory structure:")
                for root, dirs, files in os.walk(modified_crops_folder):
                    level = root.replace(modified_crops_folder, '').count(os.sep)
                    indent = '  ' * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = '  ' * (level + 1)
                    for file in sorted(files):
                        print(f"{subindent}{file}")
                
                # Test track folder extraction for dataset creation
                print(f"\nTesting track folder extraction for dataset creation:")
                for root, dirs, files in os.walk(modified_crops_folder):
                    for file in files:
                        if file.endswith('.jpg'):
                            full_path = os.path.join(root, file)
                            # Get relative path from modified_crops_folder
                            rel_path_from_modified = os.path.relpath(full_path, modified_crops_folder)
                            
                            # Extract track folder from path
                            path_parts = rel_path_from_modified.split('/')
                            if len(path_parts) >= 2:
                                track_folder = f"{path_parts[0]}_{path_parts[1]}"  # e.g., "frame123_crop_456"
                                print(f"  {rel_path_from_modified} -> track: {track_folder}")

def test_multiple_crops():
    """Test with multiple crops from different frames and trackers."""
    
    print(f"\n=== Testing Multiple Crops ===")
    
    # Simulate multiple crops
    test_crops = [
        ("frame_10/10_100_0.950.jpg", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
        ("frame_10/10_200_0.875.jpg", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
        ("frame_25/25_100_0.890.jpg", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
        ("frame_25/25_300_0.920.jpg", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        modified_crops_folder = f"{temp_dir}/crops/modified"
        
        total_images = 0
        track_folders = set()
        
        for rel_path, crop_img in test_crops:
            # Parse crop information
            path_parts = rel_path.split('/')
            if len(path_parts) >= 2:
                frame_folder = path_parts[0]
                crop_filename = path_parts[1]
                
                name_parts = crop_filename.split('_')
                if len(name_parts) >= 2:
                    frame_id = name_parts[0]
                    tracker_id = name_parts[1]
                    
                    # Generate augmented images
                    augmented_images = augment_images([crop_img])
                    
                    # Save all images (original + augmented)
                    for i, aug_img in enumerate(augmented_images):
                        if i == 0:
                            storage_path = f"{modified_crops_folder}/frame{frame_id}/crop_{tracker_id}/crop.jpg"
                        else:
                            storage_path = f"{modified_crops_folder}/frame{frame_id}/crop_{tracker_id}/crop_aug{i}.jpg"
                        
                        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
                        aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(storage_path, aug_bgr)
                        total_images += 1
                        
                        # Track folder for dataset creation
                        track_folder = f"frame{frame_id}_crop_{tracker_id}"
                        track_folders.add(track_folder)
        
        print(f"Created {total_images} total images from {len(test_crops)} source crops")
        print(f"Generated {len(track_folders)} unique track folders: {sorted(track_folders)}")
        
        # Show final structure
        print(f"\nFinal structure:")
        for root, dirs, files in os.walk(modified_crops_folder):
            level = root.replace(modified_crops_folder, '').count(os.sep)
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/ ({len(files)} files)")

if __name__ == "__main__":
    test_augmentation_folder_structure()
    test_multiple_crops()
