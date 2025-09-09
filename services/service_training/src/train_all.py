#!/usr/bin/env python3
"""
Legacy training script for LaxAI project.

⚠️  DEPRECATED: This module is deprecated and will be removed in a future version.
   Please use the new modular approach:
   - CLI: python -m cli.train_cli
   - API: from workflows.training_workflow import TrainingWorkflow
   - Programmatic: from workflows.training_workflow import train_workflow

This script maintains backward compatibility but delegates to the new workflow system.
"""
import argparse
import json
import logging
import os
import warnings
from typing import Optional

from config.logging_config import print_banner
from parameter_registry import parameter_registry
from utils.cpu_memory import clear_cpu_memory, log_comprehensive_memory_stats
# Absolute imports
from utils.env_secrets import setup_environment_secrets
from workflows.training_workflow import TrainingWorkflow

# Enable MPS fallback for unsupported operations, as recommended by PyTorch.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Setup environment
setup_environment_secrets()

# --- Configure Logging ---
logger = logging.getLogger(__name__)

# Deprecation warning
warnings.warn(
    "train_all.py is deprecated. Use 'python -m cli.train_cli' for CLI or "
    "'from workflows.training_workflow import TrainingWorkflow' for programmatic access.",
    DeprecationWarning,
    stacklevel=2
)


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
    Legacy training function - DEPRECATED.

    This function maintains backward compatibility but delegates to the new
    TrainingWorkflow class. New code should use TrainingWorkflow directly.

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

    Returns:
        Dictionary containing training results (for backward compatibility).
    """
    warnings.warn(
        "train_all.train() is deprecated. Use TrainingWorkflow class instead.",
        DeprecationWarning,
        stacklevel=2
    )

    logger.info(f"--- Starting Legacy Training Workflow for Tenant: {tenant_id} ---")

    try:
        # Create and execute new workflow
        workflow = TrainingWorkflow(
            tenant_id=tenant_id,
            verbose=verbose,
            save_intermediate=save_intermediate,
            custom_name=custom_name,
            resume_from_checkpoint=resume_from_checkpoint,
            wandb_tags=wandb_tags,
            training_kwargs=training_kwargs,
            model_kwargs=model_kwargs,
            pipeline_name=pipeline_name,
            n_datasets_to_use=n_datasets_to_use,
            eval_kwargs=eval_kwargs
        )

        result = workflow.execute()

        # Convert new result format to legacy format for backward compatibility
        legacy_result = {
            "status": result["status"],
            "datasets_found": result["datasets_found"],
            "successful_runs": result["successful_runs"],
            "total_runs": result["total_runs"],
            "custom_name": result["custom_name"],
            "training_results": result["training_results"]
        }

        # Add legacy fields if they exist in any training result
        for training_result in result["training_results"]:
            if "result" in training_result:
                # Copy pipeline result to top level for backward compatibility
                legacy_result.update(training_result["result"])
                break

        logger.info("--- Legacy Training Workflow Finished ---")

        # Clean up memory after workflow completion
        log_comprehensive_memory_stats("Legacy workflow completion")
        clear_cpu_memory(force=True)

        return legacy_result

    except Exception as e:
        logger.error(f"Error occurred during legacy workflow: {e}")

        # Clean up memory even on failure
        clear_cpu_memory(force=True)

        # Surface exceptions to callers/tests
        raise



def main():
    """
    Legacy CLI entry point - DEPRECATED.

    This function redirects to the new CLI module for backward compatibility.
    """
    warnings.warn(
        "train_all.py CLI is deprecated. Use 'python -m cli.train_cli' instead.",
        DeprecationWarning,
        stacklevel=2
    )

    print("⚠️  This CLI is deprecated. Please use: python -m cli.train_cli")
    print("Redirecting to new CLI...")

    # Import and delegate to new CLI
    try:
        from cli.train_cli import main as new_main
        new_main()
    except ImportError as e:
        print(f"❌ Failed to import new CLI: {e}")
        print("Falling back to legacy CLI...")

        # Fallback to legacy CLI if new one fails
        _legacy_main()


def _legacy_main():
    """
    Fallback legacy CLI implementation.
    """
    print_banner()

    # Create base parser with description
    parser = argparse.ArgumentParser(description="Run the full LaxAI Data Prep and Training Workflow (LEGACY).")

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