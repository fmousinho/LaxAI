#!/usr/bin/env python3
"""
End-to-end workflow script for the LaxAI project using the unverified tracks pipeline.

This script automates the following process:
1. Finds all video files in a specified tenant's 'raw' directory in GCS.
2. For each video, it runs the Unverified Tracks Pipeline.
3. Upon successful data preparation, it identifies the generated training datasets.
4. For each training dataset, it runs the Model Training Pipeline.
"""
import argparse
import logging
import os

from config import logging_config

from shared_libs.utils.env_secrets import setup_environment_secrets

setup_environment_secrets()

# IMPORTANT: Load environment variables and credentials FIRST
# This must be imported before any modules that use GCS or WandB

from config.all_config import detection_config
from unverified_track_generator_pipeline import TrackGeneratorPipeline

# Imports using relative imports since we're now in the src package
from common.google_storage import GCSPaths, get_storage

# Enable MPS fallback for unsupported operations, as recommended by PyTorch.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Configure Logging ---
# Note: This script assumes logging is configured elsewhere (e.g., in config)
# If not, uncomment the following lines for basic logging.
# from config import logging_config
logger = logging.getLogger(__name__)



def create_unverified_tracks(tenant_id: str, crop_sampling_rate: int, verbose: bool, save_intermediate: bool, resume_from_checkpoint: bool = True):
    """
    Main function to orchestrate the data prep and training workflows using the unverified tracks pipeline.

    Args:
        tenant_id: The tenant ID for GCS operations.
        crop_sampling_rate: Save a crop every N frames.
        verbose: Enable verbose logging for pipelines.
        save_intermediate: Save intermediate pipeline results to GCS.
        resume_from_checkpoint: Whether to resume from existing checkpoint data (skip completed steps).
    """
    logger.info(f"--- Starting End-to-End Workflow for Tenant: {tenant_id} ---")
    
    # Log resume configuration for user clarity
    if resume_from_checkpoint:
        logger.info("✓ Resume mode: ENABLED - Will skip completed pipeline steps")
    else:
        logger.info("⚠ Resume mode: DISABLED - Will start from scratch and overwrite existing data")

    # 1. Find all videos in the raw directory
    try:
        tenant_storage = get_storage(tenant_id)
        path_manager = GCSPaths()
        
        # Get the raw data prefix for this tenant
        raw_prefix = path_manager.get_path("raw_data")
        
        # First try the standard prefix pattern
        raw_blobs = tenant_storage.list_blobs(prefix=raw_prefix)
        
        # If no blobs found, try alternative patterns
        if not raw_blobs:
            logger.info(f"No blobs found with '{raw_prefix}' prefix, trying alternative patterns...")
            for prefix in ["/raw/", "raw", "/raw"]:
                try:
                    raw_blobs = tenant_storage.list_blobs(prefix=prefix)
                    if raw_blobs:
                        logger.info(f"Found blobs with prefix '{prefix}'")
                        break
                except Exception as prefix_error:
                    logger.warning(f"Error with prefix '{prefix}': {prefix_error}")
        
        
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
            if not resume_from_checkpoint:
                logger.error("No video files found in 'raw/' directory and resume is disabled. Exiting.")
                return
            else:
                logger.warning("No video files found in 'raw/' directory. This is expected when resuming - videos were already imported.")
                logger.info("Attempting to discover videos from existing pipeline runs...")
                
                # When resuming, try to find videos that were already imported using structured paths
                try:
                    # Look for imported videos using the process folder structure
                    process_prefix = "process/"  # Base process folder prefix
                    process_blobs = tenant_storage.list_blobs(prefix=process_prefix)
                    imported_videos = set()
                    
                    for blob in process_blobs:
                        # Look for video files in imported directories (process/{video_id}/imported/)
                        if (blob.lower().endswith(('.mp4', '.mov', '.avi')) and 
                            '/imported/' in blob):
                            filename = blob.split('/')[-1]
                            imported_videos.add(filename)
                            # For resume, we use the imported video path
                            video_paths[filename] = blob
                    
                    video_files = list(imported_videos)
                    if video_files:
                        logger.info(f"Found {len(video_files)} previously imported videos to resume: {video_files}")
                    else:
                        logger.error("No imported videos found for resume. Nothing to process.")
                        return
                        
                except Exception as e:
                    logger.error(f"Failed to discover imported videos for resume: {e}")
                    return
        else:
            logger.info(f"Found {len(video_files)} videos to process: {video_files}")
    except Exception as e:
        logger.error(f"Failed to list videos from GCS. Ensure credentials are correct. Error: {e}")
        return

    # 2. Run Unverified Tracks Pipeline for each video
    for video_file in video_files:
        logger.info(f"\n--- Processing Video: {video_file} ---")
        
        # Create a new unverified tracks pipeline instance for each video to avoid state conflicts
        # Update the config with the specified frames per video
        original_frames_per_video = detection_config.frames_per_video
        detection_config.frames_per_video = crop_sampling_rate
        
        unverified_tracks_pipeline = TrackGeneratorPipeline(
            config=detection_config,
            tenant_id=tenant_id,
            verbose=verbose,
            save_intermediate=save_intermediate
        )
        
        # Use the full path for the video
        video_full_path = video_paths[video_file]
        
        # Log resume status for user clarity
        resume_status = "with checkpoint resume enabled" if resume_from_checkpoint else "from scratch (no resume)"
        logger.info(f"Starting pipeline for {video_file} {resume_status}")
        
        dataprep_results = unverified_tracks_pipeline.run(
            video_path=video_full_path, 
            resume_from_checkpoint=resume_from_checkpoint
        )

        if dataprep_results.get("status") == "completed":
            logger.info(f"Unverified tracks pipeline completed successfully for {video_file}!")
            
        else:
            logger.error(f"Unverified tracks pipeline failed for {video_file}. Skipping training.")
            logger.error(f"Details: {dataprep_results.get('errors', [])}")
        
        # Reset the config
        detection_config.frames_per_video = original_frames_per_video

    logger.info("--- End-to-End Workflow Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Run the full LaxAI Data Prep and Training Workflow using the unverified tracks pipeline.")
    parser.add_argument("--tenant_id", type=str, default="tenant1", help="The tenant ID for GCS.")
    parser.add_argument("--crop_sampling_rate", type=int, default=20, help="Save a crop every N frames.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose pipeline logging.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate pipeline step results to GCS.")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch instead of resuming from checkpoint (default: resume enabled).")
    args = parser.parse_args()

    # Resume is enabled by default, unless --no-resume is specified
    resume_from_checkpoint = not args.no_resume

    create_unverified_tracks(
        tenant_id=args.tenant_id,
        crop_sampling_rate=args.crop_sampling_rate,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate,
        resume_from_checkpoint=resume_from_checkpoint
    )

if __name__ == "__main__":
    main()