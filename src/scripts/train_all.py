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
import logging 
import json
import argparse
from typing import Optional

# Imports using relative imports since we're now in the src package
from utils.env_secrets import setup_environment_secrets
setup_environment_secrets()
from config.logging_config import print_banner
from config.parameter_registry import parameter_registry
from common.google_storage import get_storage, GCSPaths
from train.train_pipeline import TrainPipeline

# Enable MPS fallback for unsupported operations, as recommended by PyTorch.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Configure Logging ---
# Note: This script assumes logging is configured elsewhere (e.g., in config)
# If not, uncomment the following lines for basic logging.
# from config import logging_config
logger = logging.getLogger(__name__)


def train(tenant_id: str, 
          verbose: bool = True,
          save_intermediate: bool = True,
          custom_name: str = "train_all_run",
          resume_from_checkpoint: bool = True,
          wandb_tags: Optional[list] = None,
          training_kwargs: Optional[dict] = None,
          model_kwargs: Optional[dict] = None,
          pipeline_name: Optional[str] = "default",
          n_datasets_to_use: Optional[int] = None,
          eval_kwargs: Optional[dict] = None):
    """
    Main function to orchestrate the data prep and training workflows.

    Args:
        tenant_id: The tenant ID for GCS operations.
        verbose: Enable verbose logging for pipelines.
        save_intermediate: Save intermediate pipeline results to GCS.
        custom_name: Custom name for the training run (used in wandb and logging).
        resume_from_checkpoint: Resume training from checkpoint if available.
        wandb_tags: List of tags for wandb tracking.
        training_kwargs: Dictionary of training parameters to pass to TrainPipeline.
        model_kwargs: Dictionary of model parameters to pass to model constructor.
        pipeline_name: Unique name for the pipeline (used for cancellation).
        eval_kwargs: Dictionary of evaluation parameters to pass to evaluation pipeline.
    """
    if wandb_tags is None:
        wandb_tags = []
    if training_kwargs is None:
        training_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if eval_kwargs is None:
        eval_kwargs = {}

    logger.info(f"--- Starting End-to-End Workflow for Tenant: {tenant_id} ---")
    logger.info(f"Training configuration: {training_kwargs}")
    logger.info(f"Model configuration: {model_kwargs}")

    # 1. Find all videos in the raw directory
    try:
        # Combine training_kwargs and model_kwargs for TrainPipeline
        all_kwargs = {**training_kwargs, **model_kwargs, **eval_kwargs}

        # Prevent duplicate pipeline_name if callers included it in
        # training_kwargs (tests sometimes pass pipeline_name there).
        # TrainPipeline receives pipeline_name explicitly below.
        if 'pipeline_name' in all_kwargs:
            logger.debug("Removing duplicate 'pipeline_name' from merged kwargs to avoid TypeError")
            all_kwargs.pop('pipeline_name')
        
        # Allow callers (API) to provide a pipeline_name to register the pipeline
        
        train_pipeline = TrainPipeline(
            tenant_id=tenant_id, 
            verbose=verbose, 
            save_intermediate=save_intermediate,
            pipeline_name=pipeline_name,
            **all_kwargs
        )

        logger.info("Checking for available datasets..")
        # Use Google Storage functions to list directories
        storage_client = get_storage(tenant_id)
       
        path_finder = GCSPaths()
        datasets_folder = path_finder.get_path("datasets_root")
        if datasets_folder is None:
            raise ValueError("datasets_root path not found in GCSPaths configuration.")
        else:
            datasets_folder = datasets_folder
        datasets = storage_client.list_blobs(prefix=datasets_folder, delimiter='/', exclude_prefix_in_return=True)
        datasets = list(datasets)  # Convert to list for easier processing

        logger.info(f"Found {len(datasets)} dataset directories in GCS.")

        # By default use all discovered datasets; tests may pass
        # `n_datasets_to_use=1` to limit scope for fast runs.
        N_DATASETS_TO_USE = n_datasets_to_use if n_datasets_to_use is not None else len(datasets)

        datasets_to_use = [dataset.rstrip('/') for dataset in datasets[0:N_DATASETS_TO_USE]]

        # Run the training pipeline with all the provided arguments
        train_results = train_pipeline.run(
            dataset_name=datasets_to_use, 
            custom_name=custom_name, 
            resume_from_checkpoint=resume_from_checkpoint
        )

        if train_results.get("status") == "completed":
            logger.info(f"Successfully completed training for dataset: {datasets_to_use}")
        else:
            logger.error(f"Training pipeline failed for dataset: {datasets_to_use}")
            logger.error(f"Details: {json.dumps(train_results.get('errors'), indent=2)}")


        logger.info("--- End-to-End Workflow Finished ---")

        # Return the pipeline results to callers (tests, API wrappers)
        return train_results

    except Exception as e:
        logger.error(f"Error occurred during workflow: {e}")
        logger.error(f"Details: {json.dumps(e.args, indent=2)}")
        # Surface exceptions to callers/tests
        raise



def main():

    print_banner()
    
    # Create base parser with description
    parser = argparse.ArgumentParser(description="Run the full LaxAI Data Prep and Training Workflow.")
    
    # Use parameter registry to add training/model arguments
    parser = parameter_registry.generate_cli_parser(parser)
    
    # Add non-training specific arguments
    parser.add_argument("--tenant_id", type=str, default="tenant1", help="The tenant ID for GCS.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to extract per video.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose pipeline logging.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate pipeline step results to GCS.")
    parser.add_argument("--custom_name", type=str, default="train_all_run", help="Custom name for the training run (used in wandb and logging).")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=True, help="Resume training from checkpoint if available.")
    parser.add_argument("--wandb_tags", nargs="*", default=[], help="List of tags for wandb tracking (space-separated).")
    parser.add_argument("--n_datasets_to_use", type=int, default=None, help="Limit number of discovered datasets to use for training (top-level param).")
    
    args = parser.parse_args()

    # Extract training and model kwargs using parameter registry
    training_kwargs = {}
    model_kwargs = {}
    
    for param_name, param_def in parameter_registry.parameters.items():
        arg_value = getattr(args, param_name, None)
        if arg_value is not None:
            # Determine if it's a model parameter by checking config_path
            if param_def.config_path.startswith('model_config'):
                model_kwargs[param_name] = arg_value
            else:  # training parameters
                # Avoid elevating n_datasets_to_use into training_kwargs; it
                # should be a top-level parameter passed directly to train().
                if param_name == 'n_datasets_to_use':
                    # already captured below from CLI arg --n_datasets_to_use
                    continue
                training_kwargs[param_name] = arg_value

    # A basic logging config is needed if not configured globally
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    train(
        tenant_id=args.tenant_id,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate,
        custom_name=args.custom_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        wandb_tags=args.wandb_tags,
    training_kwargs=training_kwargs,
    model_kwargs=model_kwargs,
    n_datasets_to_use=args.n_datasets_to_use
    )

if __name__ == "__main__":
    main()