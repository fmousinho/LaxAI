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

from src.config.all_config import detection_config, training_config
from src.config import logging_config
from src.common.google_storage import get_storage
from src.train.dataprep_pipeline import DataPrepPipeline
from src.train.train_pipeline import TrainPipeline

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


        # Use Google Storage functions to list directories
        storage_client = get_storage(tenant_id)
        # Find dataset paths - look for /datasets/ directories that contain numeric/train/ structure
        all_blobs = storage_client.list_blobs()
        logger.info(f"Total blobs found: {len(all_blobs)}")
        
        # Debug: Show sample blobs that contain /datasets/ and /train/
        train_blobs = [blob for blob in all_blobs if '/train/' in blob]
        for blob in train_blobs[:5]:
            logger.info(f"Sample blob: {blob}")

        # Find all unique parent directories containing /train
        train_paths = set()
        for blob in train_blobs:
            # Extract the full path including frame directory: .../datasets/frameN/train/
            if '/datasets/' in blob and '/train/' in blob:
                datasets_index = blob.find('/datasets/')
                train_index = blob.find('/train/', datasets_index)
                if datasets_index != -1 and train_index != -1:
                    # Include everything up to and including /train/
                    full_path = blob[:train_index + len('/train/')]
                    train_paths.add(full_path)
        logger.info(f"Found {len(train_paths)} directories containing training data.")

        total_folders = len(train_paths)
        processed_folders = 0
        for train_path in train_paths:
            # Validate that this path has the expected structure: .../datasets/frameN/train/
            if '/datasets/' not in train_path or '/datasets/frame' not in train_path or not train_path.rstrip('/').endswith('/train'):
                logger.warning(f"Skipping invalid train path structure: {train_path}")
                continue

            logger.info("**********************************************************************")
            logger.info(f"  Training round {processed_folders + 1}/{total_folders}")
            logger.info(f"Running train pipeline for dataset: {train_path}")
            logger.info("**********************************************************************")
            train_results = train_pipeline.run(dataset_path=train_path)

            if train_results.get("status") == "completed":
                logger.info(f"Successfully completed training for dataset: {train_path}")
            else:
                logger.error(f"Training pipeline failed for dataset: {train_path}")
                logger.error(f"Details: {json.dumps(train_results.get('errors'), indent=2)}")

            processed_folders += 1

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