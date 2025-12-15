#!/usr/bin/env python3
"""
Main entry point for LaxAI Service Tracking.

This script is the entry point for the Cloud Run job that executes track generation.
It parses command line arguments and runs the TrackingController.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared_libs.config.logging_config
logger = logging.getLogger(__name__)

logging.getLogger("tracker.byte_tracker").setLevel(logging.DEBUG)
logging.getLogger("tracker.matching").setLevel(logging.DEBUG)

from shared_libs.utils.env_secrets import setup_environment_secrets
setup_environment_secrets()

import argparse
import warnings

from schemas.tracking import TrackingParams
from tracking_controller import TrackingController


def main():
    """Main entry point for the tracking service."""
    parser = argparse.ArgumentParser(description='LaxAI Service Tracking')

    parser.add_argument('--video_path', required=True, help='Path to the video file that will be processed')
    parser.add_argument('--output_path', default="tracks.json", help='Path to the json file where the tracks will be saved')
    parser.add_argument('--detections_save_path', default=None, help='If provided, detections wil be saved to this path')

    parser.add_argument('--wandb_run_name', default=None,
                       help='Custom name for the track generation run')
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=True,
                       help='Resume track generation from checkpoint if available')

    # Tracking Params
    parser.add_argument('--prediction_threshold', type=float, default=None, help='Minimum confidence for detections to be considered')
    parser.add_argument('--track_activation_threshold', type=float, default=None, help='Track activation threshold')
    parser.add_argument('--lost_track_buffer', type=int, default=None, help='Number of frames to wait before removing a lost track')
    parser.add_argument('--max_match_distance', type=float, default=None, help='Max distance for tracks to be matched')
    parser.add_argument('--min_consecutive_frames', type=int, default=None, help='Minimum consecutive frames for track confirmation')

    args = parser.parse_args()

    try:
        # Create TrackingParams
        tracking_params = TrackingParams(
            resume_from_checkpoint=args.resume_from_checkpoint,
            prediction_threshold=args.prediction_threshold if args.prediction_threshold is not None else TrackingParams.model_fields['prediction_threshold'].default,
            track_activation_threshold=args.track_activation_threshold if args.track_activation_threshold is not None else TrackingParams.model_fields['track_activation_threshold'].default,
            lost_track_buffer=args.lost_track_buffer if args.lost_track_buffer is not None else TrackingParams.model_fields['lost_track_buffer'].default,
            max_match_distance=args.max_match_distance if args.max_match_distance is not None else TrackingParams.model_fields['max_match_distance'].default,
            min_consecutive_frames=args.min_consecutive_frames if args.min_consecutive_frames is not None else TrackingParams.model_fields['min_consecutive_frames'].default
        )

        # Create and run the controller
        controller = TrackingController(
            tracking_params=tracking_params,
            wandb_run_name=args.wandb_run_name
        )

        result = controller.run(args.video_path, args.output_path)

        logger.info(f"Track generation completed with result: {result}")
        return 0

    except Exception as e:
        logger.error(f"Track generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())