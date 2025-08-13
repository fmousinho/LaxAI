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
from pathlib import Path
from typing import Optional


# Imports using relative imports since we're now in the src package
from config import logging_config
from config.logging_config import print_banner
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
          model_kwargs: Optional[dict] = None):
    """
    Main function to orchestrate the data prep and training workflows.

    Args:
        tenant_id: The tenant ID for GCS operations.
        frames_per_video: Number of frames to extract per video in the data prep pipeline.
        verbose: Enable verbose logging for pipelines.
        save_intermediate: Save intermediate pipeline results to GCS.
        custom_name: Custom name for the training run (used in wandb and logging).
        resume_from_checkpoint: Resume training from checkpoint if available.
        wandb_tags: List of tags for wandb tracking.
        training_kwargs: Dictionary of training parameters to pass to TrainPipeline.
        model_kwargs: Dictionary of model parameters to pass to model constructor.
    """
    if wandb_tags is None:
        wandb_tags = []
    if training_kwargs is None:
        training_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
        
    logger.info(f"--- Starting End-to-End Workflow for Tenant: {tenant_id} ---")
    logger.info(f"Training configuration: {training_kwargs}")
    logger.info(f"Model configuration: {model_kwargs}")

    # 1. Find all videos in the raw directory
    try:
        # Combine training_kwargs and model_kwargs for TrainPipeline
        all_kwargs = {**training_kwargs, **model_kwargs}
        
        train_pipeline = TrainPipeline(
            tenant_id=tenant_id, 
            verbose=verbose, 
            save_intermediate=save_intermediate,
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
    
        N_DATASETS_TO_USE = 1

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

    except Exception as e:
        logger.error(f"Error occurred during workflow: {e}")
        logger.error(f"Details: {json.dumps(e.args, indent=2)}")



def main():

    print_banner()
    parser = argparse.ArgumentParser(description="Run the full LaxAI Data Prep and Training Workflow.")
    
    # Basic pipeline arguments
    parser.add_argument("--tenant_id", type=str, default="tenant1", help="The tenant ID for GCS.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to extract per video.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose pipeline logging.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate pipeline step results to GCS.")
    
    # Training pipeline specific arguments
    parser.add_argument("--custom_name", type=str, default="train_all_run", help="Custom name for the training run (used in wandb and logging).")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=True, help="Resume training from checkpoint if available.")
    parser.add_argument("--wandb_tags", nargs="*", default=[], help="List of tags for wandb tracking (space-separated).")
    
    # Training kwargs (passed to Training class) - no defaults here, let inner functions handle defaults
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training.")
    parser.add_argument("--force_pretraining", action="store_true", help="Force use of pretrained weights even if custom weights exist.")
    parser.add_argument("--early_stopping_patience", type=int, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--min_images_per_player", type=int, help="Minimum number of images required per player.")
    parser.add_argument("--margin", type=float, help="Triplet loss margin.")
    parser.add_argument("--weight_decay", type=float, help="L2 regularization weight decay.")
    parser.add_argument("--margin_decay_rate", type=float, help="Decay rate for triplet loss margin.")
    parser.add_argument("--margin_change_threshold", type=float, help="Threshold for margin changes in triplet loss.")
    parser.add_argument("--lr_scheduler_patience", type=int, help="Learning rate scheduler patience.")
    parser.add_argument("--lr_scheduler_factor", type=float, help="Learning rate reduction factor.")
    parser.add_argument("--lr_scheduler_min_lr", type=float, help="Minimum learning rate.")
    parser.add_argument("--num_workers", type=int, help="Number of DataLoader workers.")
    parser.add_argument("--prefetch_factor", type=int, help="DataLoader prefetch factor.")
    
    # Model kwargs (passed to model constructor) - no defaults here, let inner functions handle defaults
    parser.add_argument("--embedding_dim", type=int, help="Dimension of output embeddings.")
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate in embedding layer.")
    parser.add_argument("--use_cbam", action="store_true", help="Use CBAM attention modules in ResNet.")
    parser.add_argument("--no_cbam", action="store_true", help="Disable CBAM attention modules.")
    parser.add_argument("--attention_layers", nargs="*", help="ResNet layers to apply CBAM attention to.")
    
    args = parser.parse_args()

    # Build training_kwargs from only explicitly provided arguments
    training_kwargs = {}
    training_arg_names = [
        'num_epochs', 'batch_size', 'learning_rate', 'early_stopping_patience',
        'min_images_per_player', 'margin', 'weight_decay', 'margin_decay_rate',
        'margin_change_threshold', 'lr_scheduler_patience', 'lr_scheduler_factor',
        'lr_scheduler_min_lr', 'num_workers', 'prefetch_factor'
    ]
    
    for arg_name in training_arg_names:
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            training_kwargs[arg_name] = getattr(args, arg_name)
    
    # Handle force_pretraining flag (only add if True)
    if args.force_pretraining:
        training_kwargs['force_pretraining'] = True

    # Build model_kwargs from only explicitly provided arguments
    model_kwargs = {}
    model_arg_names = ['embedding_dim', 'dropout_rate', 'attention_layers']
    
    for arg_name in model_arg_names:
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            model_kwargs[arg_name] = getattr(args, arg_name)
    
    # Handle CBAM flags (only add if explicitly provided)
    if args.use_cbam and not args.no_cbam:
        model_kwargs['use_cbam'] = True
    elif args.no_cbam:
        model_kwargs['use_cbam'] = False
    # If neither flag is provided, let the model use its default

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
        model_kwargs=model_kwargs
    )

if __name__ == "__main__":
    main()