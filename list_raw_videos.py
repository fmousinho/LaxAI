#!/usr/bin/env python3
"""
Script to list all video files in the tenant's raw directory.
"""
import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.common.google_storage import get_storage

def list_raw_videos(tenant_id: str = "tenant1"):
    """
    List all video files in the tenant's raw directory.
    
    Args:
        tenant_id: The tenant ID to check (default: "tenant1")
    """
    print(f"Listing videos in raw directory for tenant: {tenant_id}")
    
    # Get storage client
    tenant_storage = get_storage(f"{tenant_id}/user")
    
    # List all blobs in the raw directory
    print("\nListing all files in raw/ directory:")
    raw_blobs = tenant_storage.list_blobs(prefix="raw/")
    
    video_files = []
    other_files = []
    
    for blob_name in raw_blobs:
        print(f"  - {blob_name}")
        if blob_name.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            video_files.append(blob_name)
        else:
            other_files.append(blob_name)
    
    print(f"\nVideo files found ({len(video_files)}):")
    for video in video_files:
        print(f"  - {video}")
    
    print(f"\nOther files found ({len(other_files)}):")
    for other in other_files:
        print(f"  - {other}")
    
    if not video_files:
        print("\nNo video files found in raw/ directory!")
        print("Make sure your video files are uploaded to the correct location.")
    
    return video_files

if __name__ == "__main__":
    list_raw_videos()
