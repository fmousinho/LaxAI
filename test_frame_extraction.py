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

def test_frame_extraction():
    """Test frame extraction on an existing processed video."""
    
    # Configure for test
    detection_config.frames_per_video = 2  # Just 2 frames for quick test
    
    # Get storage client
    pipeline = DataPrepPipeline(
        config=detection_config,
        tenant_id="tenant1",
        verbose=True,
        save_intermediate=False
    )
    
    # Try to find raw videos
    from src.common.google_storage import get_storage
    storage = get_storage('tenant1')
    
    # Look for raw videos first
    raw_blobs = list(storage.list_blobs(prefix='raw/'))
    raw_videos = [blob for blob in raw_blobs if '.mp4' in blob]
    
    print(f"Found {len(raw_videos)} raw videos:")
    for video in raw_videos:
        print(f"  {video}")
    
    if raw_videos:
        # Use the first raw video
        relative_path = raw_videos[0]
        print(f"Testing with raw video: {relative_path}")
    else:
        # Fallback to any video
        all_blobs = list(storage.list_blobs())
        video_blobs = [blob for blob in all_blobs if '.mp4' in blob]
        
        print(f"No raw videos found. Found {len(video_blobs)} total videos:")
        for video in video_blobs[:5]:
            print(f"  {video}")
        
        if not video_blobs:
            print("No video files found at all!")
            return
        
        # Prioritize raw videos if any exist
        raw_in_list = [v for v in video_blobs if '/raw/' in v]
        if raw_in_list:
            relative_path = raw_in_list[0]
            print(f"Testing with raw video from list: {relative_path}")
        else:
            relative_path = video_blobs[0]
            print(f"Testing with any video: {relative_path}")
    
    # Create a fake context for frame extraction
    import uuid
    video_guid = str(uuid.uuid4())
    temp_path = f"/tmp/test_video_{video_guid}.mp4"
    
    # Download the video
    if not storage.download_blob(relative_path, temp_path):
        print("Failed to download video!")
        return
    
    print(f"Downloaded video to {temp_path}")
    
    # Create context for frame extraction
    context = {
        "loaded_video": {
            "temp_path": temp_path,
            "video_guid": video_guid
        },
        "video_folder": f"test_run/video_{video_guid}"
    }
    
    # Test frame extraction
    try:
        result = pipeline._extract_frames_for_detections(context)
        print(f"Frame extraction result: {result}")
    except Exception as e:
        print(f"Frame extraction failed: {e}")
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    test_frame_extraction()
