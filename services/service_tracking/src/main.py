#!/usr/bin/env python3
"""
Main entry point for LaxAI Service Tracking.

This script is the entry point for the Cloud Run job that executes track generation.
It parses command line arguments and runs the UnverifiedTrackGenerationWorkflow.
"""

import argparse
import logging
import os
import sys
import warnings

import torch
from torch.jit import TracerWarning

# Ensure shared_libs can be imported
sys.path.insert(0, '/app')

# Ensure src directory is in path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Suppress PyTorch TracerWarnings that occur during model operations
warnings.filterwarnings(
    "ignore",
    message=r".*Converting a tensor to a Python boolean might cause the trace to be incorrect.*"
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.as_tensor results are registered as constants in the trace.*"
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.tensor results are registered as constants in the trace.*",
    category=TracerWarning
)
warnings.filterwarnings(
    "ignore",
    message=r".*TracerWarning.*torch\.tensor.*",
    category=TracerWarning
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from workflows.create_unverified_tracks import UnverifiedTrackGenerationWorkflow


def main():
    """Main entry point for the tracking service."""
    parser = argparse.ArgumentParser(description='LaxAI Service Tracking')
    parser.add_argument('--tenant_id', required=True, help='Tenant ID for GCS operations')
    parser.add_argument('--custom_name', default='track_generation_workflow_run',
                       help='Custom name for the track generation run')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging')
    parser.add_argument('--resume-from-checkpoint', action='store_true', default=True,
                       help='Resume track generation from checkpoint if available')
    parser.add_argument('--video_limit', type=int, default=None,
                       help='Maximum number of videos to process (None for all)')
    parser.add_argument('--task_id', default=None,
                       help='Task ID for tracking this run')

    args = parser.parse_args()

    logger.info(f"Starting track generation for tenant: {args.tenant_id}")
    logger.info(f"Custom name: {args.custom_name}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    logger.info(f"Video limit: {args.video_limit}")
    logger.info(f"Task ID: {args.task_id}")

    try:
        # Create and run the workflow
        workflow = UnverifiedTrackGenerationWorkflow(
            tenant_id=args.tenant_id,
            verbose=args.verbose,
            custom_name=args.custom_name,
            resume_from_checkpoint=args.resume_from_checkpoint,
            task_id=args.task_id,
            video_limit=args.video_limit
        )

        result = workflow.execute()

        logger.info(f"Track generation completed with result: {result}")
        return 0

    except Exception as e:
        logger.error(f"Track generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())