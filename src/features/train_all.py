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
from typing import Any

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Enable MPS fallback for unsupported operations, as recommended by PyTorch.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from config.all_config import detection_config, training_config, model_config
from config import logging_config
from common.google_storage import get_storage, GCSPaths
from train.train_pipeline import TrainPipeline

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
        # Prepare training kwargs before creating the pipeline
        training_kwargs: dict[str, Any] = {}
        training_kwargs["force_pretraining"] = True

        train_pipeline = TrainPipeline(
            tenant_id=tenant_id, 
            verbose=verbose, 
            save_intermediate=save_intermediate,
            **training_kwargs
            )

        logger.info("Checking for available datasets..")
        # Use Google Storage functions to list directories
        storage_client = get_storage(tenant_id)
       
        path_finder = GCSPaths()
        datasets_folder = path_finder.get_path("datasets_root")
        if datasets_folder is None:
            raise ValueError("datasets_root path not found in GCSPaths configuration.")
        else:
            datasets_folder = datasets_folder.rstrip('/')
        datasets = storage_client.list_blobs(prefix=datasets_folder, delimiter='/', exclude_prefix_in_return=True)
        datasets = list(datasets)  # Convert to list for easier processing

        logger.info(f"Found {len(datasets)} dataset directories in GCS.")
    
        N_DATASETS_TO_USE = len(datasets)

        datasets_to_use = [dataset.rstrip('/') for dataset in datasets[0:N_DATASETS_TO_USE]]


        run_name = "Default params, from Resnet18"
        wandb_tags = ["do not delete"]
        for keys, values in model_config.__dict__.items():
            config = f"{keys}={values}"
            wandb_tags.append(f"{config}")
            
        for keys, values in training_config.__dict__.items():
            config = f"{keys}={values}"
            wandb_tags.append(f"{config}")

        train_results = train_pipeline.run(dataset_name=datasets_to_use, custom_name=run_name, wandb_run_tags=wandb_tags)

        if train_results.get("status") == "completed":
            logger.info(f"Successfully completed training for dataset: {datasets_to_use}")
        else:
            logger.error(f"Training pipeline failed for dataset: {datasets_to_use}")
            logger.error(f"Details: {json.dumps(train_results.get('errors'), indent=2)}")


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