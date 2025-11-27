#!/usr/bin/env python3
"""
Command Line Interface for LaxAI Training.

This module provides the CLI interface for running training workflows.
"""
import logging
from shared_libs.config import logging_config
logger = logging.getLogger(__name__)

import argparse
import difflib
import json
import signal
import sys
import threading
import time
from typing import Optional, List, Dict, Any

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/services/service_training/src')
from shared_libs.utils.env_secrets import setup_environment_secrets
setup_environment_secrets()


from shared_libs.config.all_config import training_config
from workflows.training_workflow import TrainingWorkflow
from schemas.training import TrainingParams, EvalParams, ModelParams

# Global cancellation event for signal handling
cancellation_event = threading.Event()
# Global flag to track if cancellation was due to timeout
timeout_triggered = threading.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals by setting cancellation event."""
    logger.info(f"Received signal {signum}. Requesting training cancellation...")
    cancellation_event.set()


def timeout_handler():
    """Handle execution timeout by triggering auto-resume sequence."""
    logger.warning("Execution timeout reached. Initiating graceful shutdown for auto-resume...")
    timeout_triggered.set()
    cancellation_event.set()


def print_banner():
    """Print the LaxAI training banner."""
    banner = """
    ╔═══════════════════════════════════════╗
    ║      LaxAI Training Workflow CLI      ║
    ╚═══════════════════════════════════════╝
    """
    print(banner)


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
            # Basic run with default parameters
            python -m cli.train_cli --tenant-id tenant1 --custom-name my_run
            
            # With custom training parameters
            python -m cli.train_cli --tenant-id tenant1 \\
                --training-params '{"num_epochs": 100, "batch_size": 32, "learning_rate": 0.001}'
            
            # With all parameter types
            python -m cli.train_cli --tenant-id tenant1 \\
                --training-params '{"num_epochs": 50}' \\
                --eval-params '{"batch_size": 64}'
            """
    )

    # Schema-based parameters (JSON strings)
    parser.add_argument(
        '--training-params',
        type=str,
        default=None,
        help='Training parameters as JSON string (e.g., \'{"num_epochs": 100, "batch_size": 32}\')',
        dest='training_params_json'
    )
    
    parser.add_argument(
        '--eval-params',
        type=str,
        default=None,
        help='Evaluation parameters as JSON string (e.g., \'{"batch_size": 64}\')',
        dest='eval_params_json'
    )

    # Add workflow-specific arguments
    parser.add_argument(
        "--tenant-id",
        type=str,
        default="tenant1",
        help="The tenant ID for GCS operations.",
        dest="tenant_id"
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="cli_training_run",
        help="Custom name for the training run (used in wandb and logging).",
        dest="wandb_run_name"
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


def parse_json_params(json_str: Optional[str], schema_class, param_name: str):
    """
    Parse and validate JSON string against a Pydantic schema.
    
    Args:
        json_str: JSON string to parse (can be None)
        schema_class: Pydantic model class for validation
        param_name: Name of parameter for error messages
        
    Returns:
        Validated schema instance or None if json_str is None
        
    Raises:
        SystemExit: If JSON is invalid or fails validation
    """
    if json_str is None:
        return None
        
    try:
        json_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing {param_name}: Invalid JSON - {e}")
        sys.exit(1)
    
    try:
        return schema_class(**json_dict)
    except Exception as e:
        logger.error(f"Error validating {param_name} against schema: {e}")
        sys.exit(1)


def parse_args_to_workflow_kwargs(args: argparse.Namespace) -> dict:
    """
    Parse CLI arguments into workflow keyword arguments using schema validation.

    Args:
        args: Parsed command line arguments.

    Returns:
        Dictionary of keyword arguments for TrainingWorkflow.
    """
    # Parse and validate JSON parameters
    training_params = parse_json_params(
        args.training_params_json, 
        TrainingParams, 
        "--training-params"
    )
    
    eval_params = parse_json_params(
        args.eval_params_json,
        EvalParams,
        "--eval-params"
    )
    
    # Build workflow kwargs - pass Pydantic objects directly
    workflow_kwargs = {
        'tenant_id': args.tenant_id,
        'wandb_run_name': args.wandb_run_name,
        'training_params': training_params,  # Pass Pydantic object or None
        'eval_params': eval_params,          # Pass Pydantic object or None
        'task_id': args.task_id,
        'auto_resume_count': args.auto_resume_count
    }

    return workflow_kwargs






def main():
    """
    Main CLI entry point.
    """
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        parser = create_parser()
        args = parser.parse_args()  # Will error with helpful message on unknown args

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
            logger.info(f"Execution timeout set to {execution_timeout_minutes} minutes")

        # Parse arguments and create workflow
        workflow_kwargs = parse_args_to_workflow_kwargs(args)
        
        # Pass the timeout flag to workflow
        workflow_kwargs['timeout_triggered'] = timeout_triggered

        logger.info(f"Starting LaxAI Training Workflow for tenant: {args.tenant_id}")
        logger.info(f"WandB run name: {args.wandb_run_name}")
        if args.auto_resume_count > 0:
            logger.info(f"Auto-resume attempt #{args.auto_resume_count}")

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
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Training workflow failed: {e}")
        logging.exception("Training workflow error")
        sys.exit(1)


if __name__ == "__main__":
    main()