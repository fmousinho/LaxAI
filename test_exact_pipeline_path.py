#!/usr/bin/env python3
"""
Test to debug the exact path construction used in crop extraction.
This mimics the EXACT pipeline logic to see where the disconnection occurs.
"""

import sys
import os
import logging

# Add the project root to the Python path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

from src.common.google_storage import get_storage

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_exact_pipeline_path():
    """Test the exact path construction used in the pipeline."""
    
    # Mimic exact pipeline setup
    tenant_id = "tenant1"
    video_guid = "test_video"
    run_folder = f"training_runs/run_12345"  # This matches what we saw in the logs
    
    # Create storage client (same as pipeline)
    storage_client = get_storage(tenant_id)
    
    # Exact path construction from pipeline
    video_folder = f"{run_folder}/video_{video_guid}"
    frames_folder = f"{video_folder}/selected_frames"
    
    print(f"🔍 Pipeline path construction:")
    print(f"   tenant_id: {tenant_id}")
    print(f"   video_guid: {video_guid}")
    print(f"   run_folder: {run_folder}")
    print(f"   video_folder: {video_folder}")
    print(f"   frames_folder: {frames_folder}")
    
    # Exact search logic from pipeline (line 577-580)
    search_prefix = frames_folder.rstrip('/')
    print(f"   search_prefix: {search_prefix}")
    
    final_prefix = f"{search_prefix}/"
    print(f"   final_prefix: {final_prefix}")
    
    # Check what the storage client would actually search for
    print(f"\n🔍 What GoogleStorageClient will search for:")
    if hasattr(storage_client, 'config') and hasattr(storage_client.config, 'user_path'):
        print(f"   config.user_path: {storage_client.config.user_path}")
        full_search_path = f"{storage_client.config.user_path}/{final_prefix}"
        print(f"   full_search_path: {full_search_path}")
    
    # Now search using EXACT pipeline logic
    print(f"\n🔍 Searching using exact pipeline logic...")
    selected_frames_blobs = storage_client.list_blobs(prefix=final_prefix)
    print(f"   Found {len(selected_frames_blobs)} blobs")
    
    if selected_frames_blobs:
        print(f"   Sample blob names: {selected_frames_blobs[:3]}")
    
    # Also try the path that we know works from our previous test
    print(f"\n🔍 Searching using known working path...")
    working_prefix = "training_runs/run_12345/video_test_video/selected_frames/"
    working_blobs = storage_client.list_blobs(prefix=working_prefix)
    print(f"   Found {len(working_blobs)} blobs with working prefix")
    
    if working_blobs:
        print(f"   Sample working blob names: {working_blobs[:3]}")
    
    # Check if paths are different
    if final_prefix != working_prefix:
        print(f"\n❌ PATH MISMATCH DETECTED!")
        print(f"   Pipeline uses: {final_prefix}")
        print(f"   Working path:  {working_prefix}")
    else:
        print(f"\n✅ Paths match - issue must be elsewhere")

if __name__ == "__main__":
    test_exact_pipeline_path()
