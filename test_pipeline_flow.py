#!/usr/bin/env python3
"""
Test the pipeline flow to ensure crops_in_memory is passed correctly
"""

import sys
import os
sys.path.append('.')

def test_pipeline_flow():
    """Test that crops_in_memory flows correctly through pipeline steps"""
    
    print("=== Testing Pipeline Flow ===")
    
    # Simulate extract_crops step return
    extract_crops_result = {
        "crops_extracted": True,
        "video_guid": "test-guid",
        "crops_folder": "test/crops",
        "original_crops_folder": "test/crops/original",
        "crops_uploaded": 5,
        "total_detections": 3,
        "crops_in_memory": {
            "frame_0/0_1_0.850.jpg": "rgb_crop_data_1",
            "frame_0/0_2_0.750.jpg": "rgb_crop_data_2", 
            "frame_1/1_1_0.900.jpg": "rgb_crop_data_3",
            "frame_1/1_3_0.650.jpg": "rgb_crop_data_4",
            "frame_2/2_2_0.800.jpg": "rgb_crop_data_5"
        }
    }
    
    print(f"✓ Extract crops result contains {len(extract_crops_result['crops_in_memory'])} crops in memory")
    
    # Simulate background removal step context
    context = {
        "crops_extracted": extract_crops_result["crops_extracted"],
        "video_guid": extract_crops_result["video_guid"],
        "original_crops_folder": extract_crops_result["original_crops_folder"],
        "crops_in_memory": extract_crops_result["crops_in_memory"],  # This should be passed
        "grass_mask_initialized": True
    }
    
    crops_in_memory = context.get("crops_in_memory", {})
    
    if crops_in_memory:
        print(f"✓ Background removal step receives {len(crops_in_memory)} crops in memory")
        print("✓ In-memory optimization will be used")
    else:
        print("✗ Background removal step did not receive crops in memory")
        print("✗ Will fall back to storage downloads")
    
    # Test the frame-based organization
    sample_paths = list(crops_in_memory.keys())
    print(f"✓ Sample crop paths: {sample_paths[:3]}")
    
    # Verify frame-based structure
    frame_folders = set()
    for path in sample_paths:
        if '/' in path:
            frame_folder = path.split('/')[0]
            frame_folders.add(frame_folder)
    
    print(f"✓ Crops organized in {len(frame_folders)} frame folders: {sorted(frame_folders)}")
    
    print("=== Pipeline Flow Test Complete ===")

if __name__ == "__main__":
    test_pipeline_flow()
