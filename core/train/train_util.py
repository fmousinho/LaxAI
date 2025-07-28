#!/usr/bin/env python3
"""
End-to-end workflow script for the LaxAI project.

This script automates the following process:
1. Finds all video files in a specified tenant's 'raw' directory in GCS.
2. For each video, it runs the Data Preparation Pipeline.
3. Upon successful data preparation, it identifies the generated training datasets.
4. For each training dataset, it runs the Model Training Pipeline.
"""
import os
import sys
import logging
import json
import argparse

# --- Path Setup ---
# Add the project root to the Python path to allow for absolute imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Enable MPS fallback for unsupported operations, as recommended by PyTorch.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from core.common.google_storage import get_storage
from core.train.dataprep_pipeline import DataPrepPipeline
from core.train.train_pipeline import TrainPipeline
from config.all_config import detection_config, training_config
from config import logging_config

# --- Configure Logging ---
# Note: This script assumes logging is configured elsewhere (e.g., in config)
# If not, uncomment the following lines for basic logging.
# from config import logging_config
logger = logging.getLogger(__name__)



def train(tenant_id: str, frames_per_video: int, verbose: bool, save_intermediate: bool):
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

        train_pipeline = TrainPipeline(
            tenant_id=tenant_id, 
            verbose=verbose, 
            save_intermediate=save_intermediate
            )

        train_pipeline_dir = f"{project_root}/process/train_pipeline/"  # Use Google Storage path

        # Use Google Storage functions to list directories
        storage_client = get_storage(tenant_id)
        # Find dataset paths - look for /datasets/ directories that contain numeric/train/ structure
        all_blobs = storage_client.list_blobs()
        logger.info(f"Total blobs found: {len(all_blobs)}")
        
        # Debug: Show sample blobs that contain /datasets/ and /train/
        sample_blobs = [blob for blob in all_blobs if '/datasets/' in blob and '/train/' in blob][:10]
        logger.info(f"Sample blobs with /datasets/ and /train/: {sample_blobs}")
        
        # Find all unique parent directories containing /datasets/ (up to /video_GUID/)
        parent_paths = set()
        for blob in all_blobs:
            if '/datasets/' in blob and '/train/' in blob:
                # Extract path up to just before /datasets/ (the video directory)
                datasets_index = blob.find('/datasets/')
                if datasets_index != -1:
                    parent = blob[:datasets_index + 1]  # Include the trailing slash but stop before datasets
                    parent_paths.add(parent)
        logger.info(f"Found {len(parent_paths)} parent video roots: {parent_paths}")

        total_folders = 0
        processed_folders = 0
        for video_root in parent_paths:
            # Get all blobs under this video's datasets directory
            datasets_prefix = video_root + "datasets/"
            blobs = list(storage_client.list_blobs(prefix=datasets_prefix))
            logger.info(f"Found {len(blobs)} blobs under datasets prefix: {datasets_prefix}")
            if len(blobs) > 0:
                logger.info(f"Sample blobs: {blobs[:5]}")


            # Debug: Show the relative paths after datasets/ for the first 20 blobs
            rel_paths = [blob[len(datasets_prefix):].lstrip("/") for blob in blobs[:20]]
            logger.info(f"First 20 relative paths after datasets/: {rel_paths}")

            # Find all unique numeric frame directories under /datasets/
            frame_dirs = set()
            for blob in blobs:
                rel = blob[len(datasets_prefix):].lstrip("/")
                if rel:
                    parts = rel.split("/")
                    logger.debug(f"Checking blob {blob}, after datasets/: {rel}, parts: {parts}")
                    if parts[0].isdigit():
                        frame_dirs.add(parts[0])

            if not frame_dirs:
                logger.error(f"No numeric frame directories found under datasets prefix: {datasets_prefix}")
                logger.info(f"All datasets blobs: {blobs[:10]}")
                logger.info(f"First 20 relative paths after datasets/: {rel_paths}")
                continue

            logger.info(f"Found frame directories: {sorted(frame_dirs, key=lambda x: int(x))}")

            for frame_dir in sorted(frame_dirs, key=lambda x: int(x)):
                frame_path = datasets_prefix + f"{frame_dir}/"
                logger.info(f"Running train pipeline for dataset: {frame_path}")
                frame_blobs = list(storage_client.list_blobs(prefix=frame_path))
                image_blobs = [b for b in frame_blobs if b.lower().endswith((".jpg", ".jpeg", ".png"))]
                player_dirs = set()
                player_image_count = {}
                for blob in image_blobs:
                    rel = blob[len(frame_path):].lstrip("/")
                    parts = rel.split("/")
                    if len(parts) >= 2 and parts[0] == "train" and parts[1].isdigit():
                        player_id = parts[1]
                        player_dirs.add(player_id)
                        player_image_count.setdefault(player_id, 0)
                        player_image_count[player_id] += 1
                if not player_dirs:
                    logger.error(f"Frame directory does not contain any train/{{player_id}}/image.jpg directories with images in Google Storage: {frame_path}")
                    logger.info(f"Found {len(frame_blobs)} total blobs, {len(image_blobs)} image files")
                    continue
                logger.info(f"Found {len(player_dirs)} player directories with images in frame: {frame_path}")

                train_results = train_pipeline.run(dataset_path=frame_path)

                if train_results.get("status") == "completed":
                    logger.info(f"Successfully completed training for dataset: {frame_path}")
                else:
                    logger.error(f"Training pipeline failed for dataset: {frame_path}")
                    logger.error(f"Details: {json.dumps(train_results.get('errors'), indent=2)}")

                processed_folders += 1
                logger.info(f"Completed processing frame: {frame_path} ({processed_folders})")
                total_folders += 1

        logger.info("--- End-to-End Workflow Finished ---")

    except Exception as e:
        logger.error(f"Error occurred during workflow: {e}")
        logger.error(f"Details: {json.dumps(e.args, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full LaxAI Data Prep and Training Workflow.")
    parser.add_argument("--tenant_id", type=str, default="tenant1", help="The tenant ID for GCS.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to extract per video.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose pipeline logging.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate pipeline step results to GCS.")
    args = parser.parse_args()

    # A basic logging config is needed if not configured globally
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    train(
        tenant_id=args.tenant_id,
        frames_per_video=args.frames,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate
    )