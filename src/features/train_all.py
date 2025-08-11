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



def train(tenant_id: str, 
          frames_per_video: int, 
          verbose: bool, 
          save_intermediate: bool,
          custom_name: str = "train_all_run",
          resume_from_checkpoint: bool = True,
          wandb_tags: list = None,
          training_kwargs: dict = None,
          model_kwargs: dict = None):
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
            datasets_folder = datasets_folder.rstrip('/')
        datasets = storage_client.list_blobs(prefix=datasets_folder, delimiter='/', exclude_prefix_in_return=True)
        datasets = list(datasets)  # Convert to list for easier processing

        logger.info(f"Found {len(datasets)} dataset directories in GCS.")
    
        N_DATASETS_TO_USE = len(datasets)

        datasets_to_use = [dataset.rstrip('/') for dataset in datasets[0:N_DATASETS_TO_USE]]

        # Add configuration tags to wandb_tags
        final_wandb_tags = wandb_tags.copy()
        for keys, values in model_config.__dict__.items():
            config = f"{keys}={values}"
            final_wandb_tags.append(f"{config}")
            
        for keys, values in training_config.__dict__.items():
            config = f"{keys}={values}"
            final_wandb_tags.append(f"{config}")

        # Run the training pipeline with all the provided arguments
        train_results = train_pipeline.run(
            dataset_name=datasets_to_use, 
            custom_name=custom_name, 
            wandb_run_tags=final_wandb_tags,
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


if __name__ == "__main__":
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
    
    # Training kwargs (passed to Training class)
    parser.add_argument("--num_epochs", type=int, default=training_config.num_epochs, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=training_config.batch_size, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=training_config.learning_rate, help="Learning rate for training.")
    parser.add_argument("--force_pretraining", action="store_true", help="Force use of pretrained weights even if custom weights exist.")
    parser.add_argument("--early_stopping_patience", type=int, default=training_config.early_stopping_patience, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--min_images_per_player", type=int, default=training_config.min_images_per_player, help="Minimum number of images required per player.")
    parser.add_argument("--margin", type=float, default=training_config.margin, help="Triplet loss margin.")
    parser.add_argument("--weight_decay", type=float, default=training_config.weight_decay, help="L2 regularization weight decay.")
    parser.add_argument("--margin_decay_rate", type=float, default=training_config.margin_decay_rate, help="Decay rate for triplet loss margin.")
    parser.add_argument("--margin_change_threshold", type=float, default=training_config.margin_change_threshold, help="Threshold for margin changes in triplet loss.")
    parser.add_argument("--lr_scheduler_patience", type=int, default=training_config.lr_scheduler_patience, help="Learning rate scheduler patience.")
    parser.add_argument("--lr_scheduler_factor", type=float, default=training_config.lr_scheduler_factor, help="Learning rate reduction factor.")
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=training_config.lr_scheduler_min_lr, help="Minimum learning rate.")
    parser.add_argument("--num_workers", type=int, default=training_config.num_workers, help="Number of DataLoader workers.")
    parser.add_argument("--prefetch_factor", type=int, default=training_config.prefetch_factor, help="DataLoader prefetch factor.")
    
    # Model kwargs (passed to model constructor)
    parser.add_argument("--embedding_dim", type=int, default=model_config.embedding_dim, help="Dimension of output embeddings.")
    parser.add_argument("--dropout_rate", type=float, default=model_config.dropout_rate, help="Dropout rate in embedding layer.")
    parser.add_argument("--use_cbam", action="store_true", default=True, help="Use CBAM attention modules in ResNet.")
    parser.add_argument("--no_cbam", action="store_true", help="Disable CBAM attention modules.")
    parser.add_argument("--attention_layers", nargs="*", default=['layer2', 'layer3', 'layer4'], help="ResNet layers to apply CBAM attention to.")
    
    args = parser.parse_args()

    # Handle CBAM flag logic (use_cbam is True by default, no_cbam overrides it)
    use_cbam = args.use_cbam and not args.no_cbam

    # Build training_kwargs from parsed arguments
    training_kwargs = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "force_pretraining": args.force_pretraining,
        "early_stopping_patience": args.early_stopping_patience,
        "min_images_per_player": args.min_images_per_player,
        "margin": args.margin,
        "weight_decay": args.weight_decay,
        "margin_decay_rate": args.margin_decay_rate,
        "margin_change_threshold": args.margin_change_threshold,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
    }

    # Build model_kwargs from parsed arguments  
    model_kwargs = {
        "embedding_dim": args.embedding_dim,
        "dropout_rate": args.dropout_rate,
        "use_cbam": use_cbam,
        "attention_layers": args.attention_layers,
    }

    # A basic logging config is needed if not configured globally
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    train(
        tenant_id=args.tenant_id,
        frames_per_video=args.frames,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate,
        custom_name=args.custom_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        wandb_tags=args.wandb_tags,
        training_kwargs=training_kwargs,
        model_kwargs=model_kwargs
    )