#!/usr/bin/env python3
"""
Main entry point for LaxAI Service Tracking.

This script processes a local video file to generate object tracks using the ByteTrack algorithm,
handling detection, embedding extraction, and track association.

Usage Example:
   PYTHONPATH="services/service_tracking/src:." ./.venv/bin/python -m scripts.generate_tracks \
  --video_path "/Users/fmousinho/Library/Mobile Documents/com~apple~CloudDocs/Documents/Stella/Lacrosse/Fall Raw Videos/summaries/BBL North 2027 vs GRIT Dallas 2027 - 9-45am_summary.mp4" \
  --output_path "tracks.json"
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shared_libs.config.logging_config
logger = logging.getLogger(__name__)

logging.getLogger("tracker.tracker").setLevel(logging.DEBUG)
# logging.getLogger("tracker.matching").setLevel(logging.DEBUG)

import argparse
import warnings

from tracker.config import TrackingParams
from tracker.tracking_controller import TrackingController


def main():
    """Main entry point for the tracking service."""
    parser = argparse.ArgumentParser(description='LaxAI Service Tracking')

    parser.add_argument(
        '--video_path', required=True,
        help='Path to the local video file that will be processed'
    )
    parser.add_argument(
        '--output_path', default="tracks.json",
        help='Path to the json file where the tracks will be saved'
    )
    parser.add_argument(
        '--detections_save_path', default=None,
        help='If provided, detections wil be saved to this path'
    )
    parser.add_argument(
        '--embeddings_save_path', default=None,
        help='If provided, embeddings wil be saved to this path'
    )

    args = parser.parse_args()

    try:
        # Create TrackingParams from CLI args
        tracking_params_kwargs = {
            k: v for k, v in vars(args).items()
            if v is not None and k in TrackingParams.model_fields
        }
        tracking_params = TrackingParams(**tracking_params_kwargs)

        # Create and run the controller
        controller = TrackingController(
            tracking_params=tracking_params,
        )

        result = controller.run(args.video_path, args.output_path, args.detections_save_path, args.embeddings_save_path)

        logger.info(f"Track generation completed with result: {result}")
        return 0

    except Exception as e:
        logger.error(f"Track generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())