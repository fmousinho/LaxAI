#!/usr/bin/env python3
"""
Main entry point for LaxAI Service Tracking.

This script is the entry point for the Cloud Run job that executes track generation.
It parses command line arguments and runs the TrackingController.
"""

import argparse
import logging
import os
import sys
import warnings

import torch
from torch.jit._trace import TracerWarning

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

from schemas.tracking import TrackingParams
from tracking_controller import TrackingController


def main():
    """Main entry point for the tracking service."""
    parser = argparse.ArgumentParser(description='LaxAI Service Tracking')
    parser.add_argument('--tenant_id', required=True, help='Tenant ID for GCS operations')
    parser.add_argument('--custom_name', default='track_generation_run',
                       help='Custom name for the track generation run')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging')
    parser.add_argument('--resume-from-checkpoint', action='store_true', default=True,
                       help='Resume track generation from checkpoint if available')
    parser.add_argument('--video_limit', type=int, default=None,
                       help='Maximum number of videos to process (None for all)')
    parser.add_argument('--task_id', default=None,
                       help='Task ID for tracking this run')

    # Tracking Params
    parser.add_argument('--nms-iou-threshold', type=float, default=None, help='NMS IOU threshold')
    parser.add_argument('--prediction-threshold', type=float, default=None, help='Prediction confidence threshold')
    parser.add_argument('--model-checkpoint', type=str, default=None, help='Detection model checkpoint name')
    parser.add_argument('--track-activation-threshold', type=float, default=None, help='Track activation threshold')
    parser.add_argument('--lost-track-buffer', type=int, default=None, help='Lost track buffer size')
    parser.add_argument('--minimum-matching-threshold', type=float, default=None, help='Minimum matching threshold')
    parser.add_argument('--minimum-consecutive-frames', type=int, default=None, help='Minimum consecutive frames for track')
    parser.add_argument('--model-input-width', type=int, default=224, help='Model input width')
    parser.add_argument('--model-input-height', type=int, default=224, help='Model input height')

    args = parser.parse_args()

    logger.info(f"Starting track generation for tenant: {args.tenant_id}")
    logger.info(f"Custom name: {args.custom_name}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    logger.info(f"Video limit: {args.video_limit}")
    logger.info(f"Task ID: {args.task_id}")

    try:
        # Create TrackingParams
        tracking_params = TrackingParams(
            resume_from_checkpoint=args.resume_from_checkpoint,
            video_limit=args.video_limit,
            nms_iou_threshold=args.nms_iou_threshold,
            prediction_threshold=args.prediction_threshold if args.prediction_threshold is not None else TrackingParams.model_fields['prediction_threshold'].default,
            model_checkpoint=args.model_checkpoint if args.model_checkpoint is not None else TrackingParams.model_fields['model_checkpoint'].default,
            track_activation_threshold=args.track_activation_threshold if args.track_activation_threshold is not None else TrackingParams.model_fields['track_activation_threshold'].default,
            lost_track_buffer=args.lost_track_buffer if args.lost_track_buffer is not None else TrackingParams.model_fields['lost_track_buffer'].default,
            minimum_matching_threshold=args.minimum_matching_threshold if args.minimum_matching_threshold is not None else TrackingParams.model_fields['minimum_matching_threshold'].default,
            minimum_consecutive_frames=args.minimum_consecutive_frames if args.minimum_consecutive_frames is not None else TrackingParams.model_fields['minimum_consecutive_frames'].default,
            model_input_width=args.model_input_width,
            model_input_height=args.model_input_height
        )

        # Create and run the controller
        controller = TrackingController(
            tenant_id=args.tenant_id,
            tracking_params=tracking_params,
            custom_name=args.custom_name,
            task_id=args.task_id,
            verbose=args.verbose
        )

        result = controller.run()

        logger.info(f"Track generation completed with result: {result}")
        return 0

    except Exception as e:
        logger.error(f"Track generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())