#!/usr/bin/env python3
"""
End-to-end workflow script for the LaxAI project.

This script automates the following process:
1. Finds all video files in a specified tenant's 'raw' directory in GCS.
2. For each video, it runs the Data Preparation Pipeline.
3. Upon successful data preparation, it identifies the generated training datasets.
4. For each training dataset, it runs the Model Training Pipeline.
"""
import argparse
import logging
import os

from config.all_config import detection_config
from dataprep_pipeline import DataPrepPipeline

# Imports using relative imports since we're now in the src package
from common.google_storage import get_storage

# IMPORTANT: Load environment variables and credentials FIRST
# This must be imported before any modules that use GCS or WandB


# Enable MPS fallback for unsupported operations, as recommended by PyTorch.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Configure Logging ---
# Note: This script assumes logging is configured elsewhere (e.g., in config)
# If not, uncomment the following lines for basic logging.
# from config import logging_config
logger = logging.getLogger(__name__)



def load_transform(tenant_id: str, frames_per_video: int, verbose: bool, save_intermediate: bool):
    """
    Main function to orchestrate the data prep and training workflows.

    Args:
        tenant_id: The tenant ID for GCS operations.
        frames_per_video: Number of frames to extract per video in the data prep pipeline.
        verbose: Enable verbose logging for pipelines.
        save_intermediate: Save intermediate pipeline results to GCS.
    """
    logger.info(f"--- Starting End-to-End Workflow for Tenant: {tenant_id} ---")

    # 1. Find all videos in the raw directory
    try:
        tenant_storage = get_storage(tenant_id)
        
        # First try the standard prefix pattern
        raw_blobs = tenant_storage.list_blobs(prefix="raw/")
        
        # If no blobs found, try alternative patterns
        if not raw_blobs:
            logger.info("No blobs found with 'raw/' prefix, trying alternative patterns...")
            for prefix in ["/raw/", "raw", "/raw"]:
                try:
                    raw_blobs = tenant_storage.list_blobs(prefix=prefix)
                    if raw_blobs:
                        logger.info(f"Found blobs with prefix '{prefix}'")
                        break
                except Exception as prefix_error:
                    logger.warning(f"Error with prefix '{prefix}': {prefix_error}")
        
        # If still no blobs found, search all blobs for raw video files
        if not raw_blobs:
            logger.info("No blobs found with any raw prefix, searching all blobs...")
            all_blobs = tenant_storage.list_blobs()
            raw_blobs = [
                blob for blob in all_blobs
                if 'raw' in blob.lower() and blob.lower().endswith(('.mp4', '.mov', '.avi'))
            ]
            if raw_blobs:
                logger.info(f"Found {len(raw_blobs)} video files containing 'raw' in path")
        
        # Extract video filenames and their full paths
        video_files = []
        video_paths = {}  # Map filename to full path
        
        for blob in raw_blobs:
            if blob.lower().endswith(('.mp4', '.mov', '.avi')) and not blob.endswith('/'):
                filename = blob.split('/')[-1]
                video_files.append(filename)
                
                # Strip tenant prefix if present to get relative path
                # The blob path might be like "tenant1/user/raw/video.mp4"
                # We need to extract just "raw/video.mp4" for the pipeline
                if '/raw/' in blob:
                    relative_path = 'raw/' + filename
                else:
                    relative_path = blob
                
                video_paths[filename] = relative_path
                logger.info(f"Video: {filename} -> Path: {relative_path}")
        
        if not video_files:
            logger.warning("No video files found in 'raw/' directory. Exiting.")
            return
        logger.info(f"Found {len(video_files)} videos to process: {video_files}")
    except Exception as e:
        logger.error(f"Failed to list videos from GCS. Ensure credentials are correct. Error: {e}")
        return

    # 2. Run DataPrepPipeline for each video
    for video_file in video_files:
        logger.info(f"\n--- Processing Video: {video_file} ---")
        
        # Create a new pipeline instance for each video to avoid state conflicts
        # Update the config with the specified frames per video
        original_frames_per_video = detection_config.frames_per_video
        detection_config.frames_per_video = frames_per_video
        
        dataprep_pipeline = DataPrepPipeline(
            config=detection_config,
            tenant_id=tenant_id,
            verbose=verbose,
            save_intermediate=save_intermediate
        )
        
        # Use the full path for the video
        video_full_path = video_paths[video_file]
        dataprep_results = dataprep_pipeline.run(video_path=video_full_path)

        if dataprep_results.get("status") == "completed":
            logger.info(f"Data prep pipeline completed successfully for {video_file}!")
            
        else:
            logger.error(f"Data prep pipeline failed for {video_file}. Skipping training.")
            logger.error(f"Details: {dataprep_results.get('errors', [])}")
        
        # Reset the config
        detection_config.frames_per_video = original_frames_per_video

    logger.info("--- End-to-End Workflow Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Run the full LaxAI Data Prep and Training Workflow.")
    parser.add_argument("--tenant_id", type=str, default="tenant1", help="The tenant ID for GCS.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to extract per video.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose pipeline logging.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate pipeline step results to GCS.")
    args = parser.parse_args()

    load_transform(
        tenant_id=args.tenant_id,
        frames_per_video=args.frames,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate
    )

if __name__ == "__main__":
    main()