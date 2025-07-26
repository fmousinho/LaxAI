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


def main(tenant_id: str, frames_per_video: int, verbose: bool, save_intermediate: bool):
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
        tenant_storage = get_storage(f"{tenant_id}/user")
        raw_blobs = tenant_storage.list_blobs(prefix="raw/")
        video_files = [
            blob.split('/')[-1] for blob in raw_blobs
            if blob.lower().endswith(('.mp4', '.mov', '.avi')) and not blob.endswith('/')
        ]
        if not video_files:
            logger.warning("No video files found in 'raw/' directory. Exiting.")
            return
        logger.info(f"Found {len(video_files)} videos to process: {video_files}")
    except Exception as e:
        logger.error(f"Failed to list videos from GCS. Ensure credentials are correct. Error: {e}")
        return
    dataprep_pipeline = DataPrepPipeline(
            config=detection_config,
            tenant_id=tenant_id,
            verbose=verbose,
            save_intermediate=save_intermediate
        )
    train_pipeline = TrainPipeline(tenant_id=tenant_id, verbose=verbose, save_intermediate=save_intermediate)

    # 2. Run DataPrepPipeline for each video
    for video_file in video_files:
        logger.info(f"\n--- Processing Video: {video_file} ---")
        
        dataprep_results = dataprep_pipeline.run(video_path=video_file)

        if dataprep_results.get("status") != "completed":
            logger.error(f"Data prep pipeline failed for {video_file}. Skipping training.")
            logger.error(f"Details: {json.dumps(dataprep_results.get('errors'), indent=2)}")
            continue

        # 3. Run TrainPipeline for each resulting dataset
        datasets_folder = dataprep_results.get("context", {}).get("datasets_folder")
        if not datasets_folder:
            logger.error(f"Could not find 'datasets_folder' in dataprep results for {video_file}. Skipping training.")
            continue

        logger.info(f"Data prep successful. Starting training for datasets in: {datasets_folder}")
       
        # Assuming the train pipeline needs to run on the 'train' subdirectory of the datasets folder.
        # This part may need adjustment based on the exact structure and requirements.
        train_dataset_path = os.path.join(datasets_folder, "train") # This assumes a single train/val split.
        train_results = train_pipeline.run(dataset_path=train_dataset_path)

        if train_results.get("status") == "completed":
            logger.info(f"Successfully completed training for dataset from video {video_file}.")
        else:
            logger.error(f"Training pipeline failed for dataset from video {video_file}.")
            logger.error(f"Details: {json.dumps(train_results.get('errors'), indent=2)}")

    logger.info("--- End-to-End Workflow Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full LaxAI Data Prep and Training Workflow.")
    parser.add_argument("--tenant_id", type=str, default="tenant1", help="The tenant ID for GCS.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to extract per video.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose pipeline logging.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate pipeline step results to GCS.")
    args = parser.parse_args()

    # A basic logging config is needed if not configured globally
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main(
        tenant_id=args.tenant_id,
        frames_per_video=args.frames,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate
    )
