#!/usr/bin/env python3
"""
Command Line Interface for LaxAI Training.

This module provides the CLI interface for running training workflows.
"""
import argparse
import logging
import signal
import sys
import threading
from typing import Optional

sys.path.insert(0, '/app')
from shared_libs.config import logging_config
from shared_libs.config.logging_config import print_banner
from parameter_registry import parameter_registry
from workflows.training_workflow import TrainingWorkflow

logger = logging.getLogger(__name__)
# Absolute imports
from shared_libs.utils.env_secrets import setup_environment_secrets

# Setup environment
setup_environment_secrets()

# Global cancellation event for signal handling
cancellation_event = threading.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals by setting cancellation event."""
    print(f"\n⏹️  Received signal {signum}. Requesting training cancellation...")
    cancellation_event.set()


def create_parser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser with all training parameters.

    Returns:
        Configured ArgumentParser instance.
    """
    print_banner()

    parser = argparse.ArgumentParser(
        description="LaxAI Training CLI - Run end-to-end training workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.train_cli --tenant_id tenant1 --verbose
  python -m cli.train_cli --tenant_id tenant1 --n_datasets_to_use 2 --custom_name my_training_run
        """
    )

    # Use parameter registry to add training/model arguments
    parser = parameter_registry.generate_cli_parser(parser)

    # Add workflow-specific arguments
    parser.add_argument(
        "--tenant_id",
        type=str,
        default="tenant1",
        help="The tenant ID for GCS operations."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose pipeline logging."
    )

    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate pipeline step results to GCS."
    )

    parser.add_argument(
        "--custom_name",
        type=str,
        default="cli_training_run",
        help="Custom name for the training run (used in wandb and logging)."
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        default=True,
        help="Resume training from checkpoint if available."
    )

    parser.add_argument(
        "--wandb_tags",
        nargs="*",
        default=[],
        help="List of tags for wandb tracking (space-separated)."
    )

    parser.add_argument(
        "--n_datasets_to_use",
        type=int,
        default=None,
        help="Limit number of discovered datasets to use for training."
    )

    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help="Task ID for tracking this training run (used by proxy service)."
    )

    return parser


def parse_args_to_workflow_kwargs(args: argparse.Namespace) -> dict:
    """
    Parse CLI arguments into workflow keyword arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Dictionary of keyword arguments for TrainingWorkflow.
    """
    # Extract training, model, and eval kwargs using parameter registry
    training_kwargs = {}
    model_kwargs = {}
    eval_kwargs = {}

    for param_name, param_def in parameter_registry.parameters.items():
        arg_value = getattr(args, param_name, None)
        if arg_value is not None:
            # Determine parameter category by checking config_path
            if param_def.config_path.startswith('model_config'):
                model_kwargs[param_name] = arg_value
            elif param_def.config_path.startswith('evaluator_config'):
                # Handle parameter name mapping for eval params
                if param_name == "eval_prefetch_factor":
                    eval_kwargs["prefetch_factor"] = arg_value
                else:
                    eval_kwargs[param_name] = arg_value
            else:  # training parameters
                # Avoid elevating n_datasets_to_use into training_kwargs
                if param_name == 'n_datasets_to_use':
                    continue
                training_kwargs[param_name] = arg_value

    # Build workflow kwargs
    workflow_kwargs = {
        'tenant_id': args.tenant_id,
        'verbose': args.verbose,
        'save_intermediate': args.save_intermediate,
        'custom_name': args.custom_name,
        'resume_from_checkpoint': args.resume_from_checkpoint,
        'wandb_tags': args.wandb_tags,
        'training_kwargs': training_kwargs,
        'model_kwargs': model_kwargs,
        'eval_kwargs': eval_kwargs,
        'n_datasets_to_use': args.n_datasets_to_use,
        'task_id': args.task_id
    }

    return workflow_kwargs


def main():
    """
    Main CLI entry point.
    """
    try:
        parser = create_parser()
        args = parser.parse_args()

        # Setup basic logging if not configured globally
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Setup signal handlers for graceful cancellation
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Parse arguments and create workflow
        workflow_kwargs = parse_args_to_workflow_kwargs(args)

        print(f"🚀 Starting LaxAI Training Workflow for tenant: {args.tenant_id}")
        print(f"📊 Custom run name: {args.custom_name}")
        if args.n_datasets_to_use:
            print(f"🎯 Limiting to {args.n_datasets_to_use} datasets")

        # Execute workflow with cancellation support
        workflow_kwargs['cancellation_event'] = cancellation_event
        workflow = TrainingWorkflow(**workflow_kwargs)
        result = workflow.execute()

        # Print results
        print("\n" + "="*60)
        print("🏁 TRAINING WORKFLOW COMPLETED")
        print("="*60)
        print(f"📈 Status: {result['status']}")
        print(f"📁 Datasets found: {result['datasets_found']}")
        print(f"✅ Successful runs: {result['successful_runs']}")
        print(f"📊 Total runs: {result['total_runs']}")

        if result['training_results']:
            print("\n📋 Dataset Results:")
            for training_result in result['training_results']:
                status_icon = "✅" if training_result['status'] == 'success' else "❌"
                print(f"  {status_icon} {training_result['dataset']}: {training_result['status']}")

        print(f"\n🎯 Custom name: {result['custom_name']}")
        print("="*60)

        # Exit with appropriate code
        if result['status'] == 'completed' and result['successful_runs'] > 0:
            print("🎉 Training workflow completed successfully!")
            sys.exit(0)
        elif result['status'] == 'cancelled':
            print("⏹️  Training workflow was cancelled.")
            sys.exit(130)
        else:
            print("⚠️  Training workflow completed with issues.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Training workflow failed: {e}")
        logging.exception("Training workflow error")
        sys.exit(1)


if __name__ == "__main__":
    main()