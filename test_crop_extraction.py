#!/usr/bin/env python3

import os
import sys
import logging

# Add the project root to the Python path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

# Configure logging to see all messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(message)s')

from src.train.dataprep_pipeline import DataPrepPipeline
from src.config.all_config import detection_config

def test_crop_extraction():
    """Test crop extraction on an existing processed video."""
    
    # Configure for test
    detection_config.frames_per_video = 5  # Just 5 frames for quick test
    
    # Create a pipeline
    pipeline = DataPrepPipeline(
        config=detection_config,
        tenant_id="tenant1",
        verbose=True,
        save_intermediate=False
    )
    
    # Use an existing video from the recent run
    video_guid = "b9e10694-bbaa-40ca-bfba-05c4a2ae4e1a"
    run_guid = "a6233754-1b50-4cd4-8b0d-dbdea4503883"
    
    # Create a test context with necessary data
    context = {
        "video_guid": video_guid,
        "video_folder": f"process/train_pipeline/run_{run_guid}/video_{video_guid}",
        "selected_frames_folder": f"process/train_pipeline/run_{run_guid}/video_{video_guid}/selected_frames"
    }
    
    print(f"Testing crop extraction for video: {video_guid}")
    print(f"Using context: {context}")
    
    # Test the crop extraction step directly
    try:
        result = pipeline._extract_crops(context)
        print(f"Crop extraction result: {result}")
        if result and result.get("status") == "success":
            print("✅ Crop extraction succeeded!")
        else:
            print("❌ Crop extraction failed!")
    except Exception as e:
        print(f"❌ Crop extraction failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crop_extraction()
