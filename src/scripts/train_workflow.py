#!/usr/bin/env python3
"""
Training workflow script.

This script provides a command-line interface to the training workflow.
"""
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared_libs.utils.env_secrets import setup_environment_secrets

# Setup environment secrets
setup_environment_secrets()

# Import training workflow
from services.service_training.src.workflows.training_workflow import \
    train_workflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Workflow Script")

    parser.add_argument(
        "--tenant_id",
        type=str,
        required=True,
        help="Tenant ID for GCS operations"
    )

    parser.add_argument(
        "--custom_name",
        type=str,
        default="training_workflow_run",
        help="Custom name for the training run"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )

    parser.add_argument(
        "--n_datasets_to_use",
        type=int,
        default=None,
        help="Limit number of datasets to use"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        default=True,
        help="Save intermediate results"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available"
    )

    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=[],
        help="WandB tags for tracking"
    )

    parser.add_argument(
        "--pipeline_name",
        type=str,
        default="default",
        help="Unique name for the pipeline"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Prepare training parameters
    training_kwargs = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    # Prepare wandb tags
    wandb_tags = args.wandb_tags if args.wandb_tags else None

    logger.info(f"Starting training workflow for tenant: {args.tenant_id}")
    logger.info(f"Custom name: {args.custom_name}")
    logger.info(f"Training parameters: {training_kwargs}")

    try:
        # Execute training workflow
        result = train_workflow(
            tenant_id=args.tenant_id,
            verbose=args.verbose,
            save_intermediate=args.save_intermediate,
            custom_name=args.custom_name,
            resume_from_checkpoint=args.resume_from_checkpoint,
            wandb_tags=wandb_tags,
            training_kwargs=training_kwargs,
            pipeline_name=args.pipeline_name,
            n_datasets_to_use=args.n_datasets_to_use,
        )

        # Print results
        print("\n" + "="*50)
        print("TRAINING WORKFLOW RESULTS")
        print("="*50)
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        if result.get("status") == "completed" and result.get("successful_runs", 0) > 0:
            logger.info("Training workflow completed successfully!")
            sys.exit(0)
        else:
            logger.error("Training workflow failed or had no successful runs")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Training workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
