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



# Ensure shared_libs can be imported
sys.path.insert(0, '/app')

# Ensure src directory is in path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

from schemas.tracking import TrackingParams
from tracking_controller import TrackingController


def main():
    """Main entry point for the tracking service."""
    parser = argparse.ArgumentParser(description='LaxAI Service Tracking')

    parser.add_argument('--video_path', required=True, help='Path to the video file that will be processed')
    parser.add_argument('--output_path', default="tracks.json", help='Path to the json file where the tracks will be saved')

    parser.add_argument('--wandb_run_name', default=None,
                       help='Custom name for the track generation run')
    parser.add_argument('--resume-from-checkpoint', action='store_true', default=True,
                       help='Resume track generation from checkpoint if available')

    # Tracking Params
    parser.add_argument('--prediction-threshold', type=float, default=None, help='Prediction confidence threshold')
    parser.add_argument('--track-activation-threshold', type=float, default=None, help='Track activation threshold')
    parser.add_argument('--lost-track-buffer', type=int, default=None, help='Lost track buffer size')
    parser.add_argument('--minimum-matching-threshold', type=float, default=None, help='Minimum matching threshold')
    parser.add_argument('--minimum-consecutive-frames', type=int, default=None, help='Minimum consecutive frames for track')

    args = parser.parse_args()

    try:
        # Create TrackingParams
        tracking_params = TrackingParams(
            resume_from_checkpoint=args.resume_from_checkpoint,
            prediction_threshold=args.prediction_threshold if args.prediction_threshold is not None else TrackingParams.model_fields['prediction_threshold'].default,
            track_activation_threshold=args.track_activation_threshold if args.track_activation_threshold is not None else TrackingParams.model_fields['track_activation_threshold'].default,
            lost_track_buffer=args.lost_track_buffer if args.lost_track_buffer is not None else TrackingParams.model_fields['lost_track_buffer'].default,
            minimum_matching_threshold=args.minimum_matching_threshold if args.minimum_matching_threshold is not None else TrackingParams.model_fields['minimum_matching_threshold'].default,
            minimum_consecutive_frames=args.minimum_consecutive_frames if args.minimum_consecutive_frames is not None else TrackingParams.model_fields['minimum_consecutive_frames'].default
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