#!/usr/bin/env python3
"""
Command Line Interface for LaxAI Training.

This module provides the CLI interface for running training workflows.
"""
import argparse
import difflib
import logging
import signal
import sys
import threading
import time
from typing import Optional, List

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/services/service_training/src')
from shared_libs.utils.env_secrets import setup_environment_secrets
setup_environment_secrets()

from shared_libs.config import logging_config
from shared_libs.config.logging_config import print_banner
from shared_libs.config.all_config import training_config
from workflows.training_workflow import TrainingWorkflow

logger = logging.getLogger(__name__)

# Global cancellation event for signal handling
cancellation_event = threading.Event()
# Global flag to track if cancellation was due to timeout
timeout_triggered = threading.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals by setting cancellation event."""
    print(f"\n⏹️  Received signal {signum}. Requesting training cancellation...")
    cancellation_event.set()


def timeout_handler():
    """Handle execution timeout by triggering auto-resume sequence."""
    print(f"\n⏰ Execution timeout reached. Initiating graceful shutdown for auto-resume...")
    logger.warning("Approaching execution time limit, triggering auto-resume sequence")
    timeout_triggered.set()
    cancellation_event.set()


def create_parser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser with all training parameters.

    Returns:
        Configured ArgumentParser instance.
    """
    print_banner()

    # Create the standard ArgumentParser first
    parser = argparse.ArgumentParser(
        description="LaxAI Training CLI - Run end-to-end training workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python -m cli.train_cli --tenant_id tenant1 --verbose
            python -m cli.train_cli --tenant_id tenant1 --n_datasets_to_use 2 --custom_name my_training_run
            """
    )

    # Training parameters - explicit argparse definitions
    parser.add_argument('--num-epochs', type=int, default=training_config.num_epochs,
                       help="Number of training epochs", dest='num_epochs')
    parser.add_argument('--batch-size', type=int, default=training_config.batch_size,
                       help="Batch size for training", dest='batch_size')
    parser.add_argument('--num-workers', type=int, default=training_config.num_workers,
                       help="Number of DataLoader workers", dest='num_workers')
    parser.add_argument('--learning-rate', type=float, default=training_config.learning_rate,
                       help="Learning rate for optimizer", dest='learning_rate')
    parser.add_argument('--margin', type=float, default=training_config.margin,
                       help="Margin for triplet loss", dest='margin')
    parser.add_argument('--weight-decay', type=float, default=training_config.weight_decay,
                       help="Weight decay for optimizer", dest='weight_decay')
    parser.add_argument('--lr-scheduler-patience', type=int, default=training_config.lr_scheduler_patience,
                       help="Patience for learning rate scheduler", dest='lr_scheduler_patience')
    parser.add_argument('--lr-scheduler-threshold', type=float, default=training_config.lr_scheduler_threshold,
                       help="Threshold for learning rate scheduler", dest='lr_scheduler_threshold')
    parser.add_argument('--lr-scheduler-min-lr', type=float, default=training_config.lr_scheduler_min_lr,
                       help="Minimum learning rate for scheduler", dest='lr_scheduler_min_lr')
    parser.add_argument('--lr-scheduler-factor', type=float, default=training_config.lr_scheduler_factor,
                       help="Factor for LR scheduler", dest='lr_scheduler_factor')
    parser.add_argument('--force-pretraining', action='store_true', default=training_config.force_pretraining,
                       help="Force pretrained ResNet defaults", dest='force_pretraining')
    parser.add_argument('--early-stopping-patience', type=int, default=training_config.early_stopping_patience,
                       help="Early stopping patience", dest='early_stopping_patience')
    parser.add_argument('--min-images-per-player', type=int, default=training_config.min_images_per_player,
                       help="Minimum images per player", dest='min_images_per_player')
    parser.add_argument('--train-prefetch-factor', type=int, default=training_config.prefetch_factor,
                       help="Prefetch factor for data loading", dest='train_prefetch_factor')
    parser.add_argument('--margin-decay-rate', type=float, default=training_config.margin_decay_rate,
                       help="Margin decay rate", dest='margin_decay_rate')
    parser.add_argument('--margin-change-threshold', type=float, default=training_config.margin_change_threshold,
                       help="Margin change threshold", dest='margin_change_threshold')
    parser.add_argument('--train-ratio', type=float, default=training_config.train_ratio,
                       help="Training ratio", dest='train_ratio')
    parser.add_argument('--n-datasets-to-use', type=int, default=training_config.n_datasets_to_use,
                       help="Number of datasets to use", dest='n_datasets_to_use')
    parser.add_argument('--dataset-address', type=str, default=training_config.dataset_address,
                       help="GCS path to specific dataset", dest='dataset_address')
    parser.add_argument('--use-classification-head', action='store_true', default=training_config.use_classification_head,
                       help="Whether to use classification head", dest='use_classification_head')
    parser.add_argument('--classification-epochs', type=int, default=training_config.classification_epochs,
                       help="Number of epochs to use classification loss", dest='classification_epochs')
    parser.add_argument('--classification-weight-start', type=float, default=training_config.classification_weight_start,
                       help="Initial weight for classification loss", dest='classification_weight_start')

    # Model selection overrides
    parser.add_argument('--model-class-module', type=str, default=None,
                       help="Override the model class module (e.g., siamesenet_resnet50_wCBAM)", dest='model_class_module')
    parser.add_argument('--model-class', type=str, default=None,
                       help="Override the model class name (default: SiameseNet)", dest='model_class')

    # Add workflow-specific arguments
    parser.add_argument(
        "--tenant-id",
        type=str,
        default="tenant1",
        help="The tenant ID for GCS operations.",
        dest="tenant_id"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose pipeline logging."
    )

    parser.add_argument(
        "--custom-name",
        type=str,
        default="cli_training_run",
        help="Custom name for the training run (used in wandb and logging).",
        dest="custom_name"
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        default=True,
        help="Resume training from checkpoint if available.",
        dest="resume_from_checkpoint"
    )

    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=[],
        help="List of tags for wandb tracking (space-separated).",
        dest="wandb_tags"
    )

    # Note: dataset selection options are provided explicitly above.

    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Task ID for tracking this training run (used by proxy service).",
        dest="task_id"
    )
    
    parser.add_argument(
        "--auto-resume-count",
        type=int,
        default=0,
        help="Number of times this job has been auto-resumed (internal use).",
        dest="auto_resume_count"
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
    # Extract training, model, and eval kwargs directly from args
    training_kwargs = {}
    model_kwargs = {}
    eval_kwargs = {}

    # Training parameters
    training_param_names = [
        'num_epochs', 'batch_size', 'num_workers', 'learning_rate', 'margin',
        'weight_decay', 'lr_scheduler_patience', 'lr_scheduler_threshold',
        'lr_scheduler_min_lr', 'lr_scheduler_factor', 'force_pretraining',
        'early_stopping_patience', 'min_images_per_player', 'margin_decay_rate',
        'margin_change_threshold', 'train_ratio', 'train_prefetch_factor',
        'use_classification_head', 'classification_epochs', 'classification_weight_start'
    ]
    for param_name in training_param_names:
        arg_value = getattr(args, param_name, None)
        if arg_value is not None:
            if param_name == "train_prefetch_factor":
                training_kwargs["prefetch_factor"] = arg_value
            else:
                training_kwargs[param_name] = arg_value

    # Model parameters (passed through; TrainPipeline will consume overrides)
    if getattr(args, 'model_class_module', None):
        model_kwargs['model_class_module'] = args.model_class_module
    if getattr(args, 'model_class', None):
        # TrainPipeline expects 'model_class_str'
        model_kwargs['model_class_str'] = args.model_class

    # Build workflow kwargs
    workflow_kwargs = {
        'tenant_id': args.tenant_id,
        'verbose': args.verbose,
        'custom_name': args.custom_name,
        'resume_from_checkpoint': args.resume_from_checkpoint,
        'wandb_tags': args.wandb_tags,
        'training_kwargs': training_kwargs,
        'model_kwargs': model_kwargs,
        'eval_kwargs': eval_kwargs,
        'n_datasets_to_use': getattr(args, 'n_datasets_to_use', None),
        'dataset_address': getattr(args, 'dataset_address', None),
        'task_id': args.task_id,
        'auto_resume_count': args.auto_resume_count
    }

    return workflow_kwargs


def validate_and_suggest_args(unknown_args: List[str]) -> None:
    """
    Validate unknown arguments and provide helpful suggestions.
    
    Args:
        unknown_args: List of unrecognized command line arguments
    """
    if not unknown_args:
        return
        
    # Known parameter names
    known_params = {
        '--num-epochs', '--batch-size', '--num-workers', '--learning-rate',
        '--margin', '--weight-decay', '--lr-scheduler-patience',
        '--lr-scheduler-threshold', '--lr-scheduler-min-lr',
        '--lr-scheduler-factor', '--force-pretraining',
        '--early-stopping-patience', '--min-images-per-player',
        '--train-prefetch-factor', '--margin-decay-rate',
        '--margin-change-threshold', '--train-ratio',
        '--n-datasets-to-use', '--dataset-address',
        '--tenant-id', '--verbose', '--custom-name',
        '--resume-from-checkpoint', '--wandb-tags', '--task-id',
        '--auto-resume-count', '--model-class-module', '--model-class'
    }
    
    for unknown_arg in unknown_args:
        if unknown_arg.startswith('--'):
            arg_name = unknown_arg.split('=')[0]  # Handle --arg=value format
            
            # Look for close matches
            from difflib import get_close_matches
            suggestions = get_close_matches(arg_name, known_params, n=3, cutoff=0.6)
            
            if suggestions:
                logger.warning(f"Unknown argument '{arg_name}'. Did you mean: {', '.join(suggestions)}?")
            else:
                # Check for common mistakes
                if 'prefetch' in arg_name.lower():
                    logger.warning(f"Unknown argument '{arg_name}'. Note: use --train-prefetch-factor for training or --eval-prefetch-factor for evaluation.")
                else:
                    logger.warning(f"Unknown argument '{arg_name}'. Check the parameter registry for available options.")


def main():
    """
    Main CLI entry point.
    """
    try:
        # Print all received arguments for debugging
        print(f"🔍 Received CLI arguments: {sys.argv[1:]}")
        
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        parser = create_parser()
        
        # Use parse_known_args to handle unrecognized arguments gracefully
        args, unknown_args = parser.parse_known_args()
        
        # Debug: Print what was parsed
        print(f"✅ Parsed known args: {len(vars(args))} arguments")
        print(f"⚠️  Unknown args: {unknown_args}")
        
        # Debug: Show a few of the parsed arguments
        parsed_args = vars(args)
        print(f"🔍 Sample parsed args: {list(parsed_args.keys())[:10]}...")
        
        # Warn about unrecognized arguments
        if unknown_args:
            logger.warning(f"Unrecognized arguments detected: {' '.join(unknown_args)}")
            logger.warning("These arguments will be ignored. Please check for typos or missing parameter definitions.")
            logger.info("Continuing with execution using recognized arguments only...")

        # Setup basic logging if not configured globally
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Setup signal handlers for graceful cancellation
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Setup execution timeout timer for auto-resume
        execution_timeout_minutes = training_config.execution_timeout_minutes
        if execution_timeout_minutes and execution_timeout_minutes > 0:
            timeout_seconds = execution_timeout_minutes * 60
            timeout_timer = threading.Timer(timeout_seconds, timeout_handler)
            timeout_timer.daemon = True
            timeout_timer.start()
            logger.info(f"⏰ Execution timeout set to {execution_timeout_minutes} minutes")
            print(f"⏰ Auto-resume will trigger after {execution_timeout_minutes} minutes")

        # Parse arguments and create workflow
        workflow_kwargs = parse_args_to_workflow_kwargs(args)
        
        # Pass the timeout flag to workflow
        workflow_kwargs['timeout_triggered'] = timeout_triggered

        print(f"🚀 Starting LaxAI Training Workflow for tenant: {args.tenant_id}")
        print(f"📊 Custom run name: {args.custom_name}")
        if args.dataset_address:
            print(f"📍 Using specific dataset: {args.dataset_address}")
        elif args.n_datasets_to_use:
            print(f"🎯 Limiting to {args.n_datasets_to_use} datasets")
        if args.auto_resume_count > 0:
            print(f"🔄 Auto-resume attempt #{args.auto_resume_count}")
            logger.info(f"This is auto-resume attempt #{args.auto_resume_count}")

        # Execute workflow with cancellation support
        workflow_kwargs['cancellation_event'] = cancellation_event
        workflow = TrainingWorkflow(**workflow_kwargs)
        result = workflow.execute()

        # Print results
        print("\n" + "="*60)
        print("🏁 TRAINING WORKFLOW COMPLETED")
        print("="*60)
        print(f"📈 Status: {result.get('status', 'unknown')}")
        print(f"📁 Datasets found: {result.get('datasets_found', 0)}")
        print(f"✅ Successful runs: {result.get('successful_runs', 0)}")
        print(f"📊 Total runs: {result.get('total_runs', 0)}")

        training_results = result.get('training_results') or []
        if training_results:
            print("\n📋 Dataset Results:")
            for training_result in training_results:
                status_icon = "✅" if training_result['status'] == 'success' else "❌"
                print(f"  {status_icon} {training_result['dataset']}: {training_result['status']}")

        print(f"\n🎯 Custom name: {result.get('custom_name', '')}")
        print("="*60)

        # Exit with appropriate code
        if result.get('status') == 'completed' and result.get('successful_runs', 0) > 0:
            print("🎉 Training workflow completed successfully!")
            sys.exit(0)
        elif result.get('status') == 'cancelled':
            print("⏹️  Training workflow was cancelled.")
            sys.exit(130)
        elif result.get('status') == 'auto_suspended':
            print("⏸️  Training workflow auto-suspended for resume (reached time limit).")
            sys.exit(0)  # Exit cleanly to allow auto-resume
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