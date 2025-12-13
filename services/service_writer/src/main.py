#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
import sys
import warnings

from writer import Writer


def main():
    """Main entry point for the tracking service."""
    parser = argparse.ArgumentParser(description='LaxAI Service Writing')

    parser.add_argument('--video_path', required=True, help='Path to the video file that will be processed')
    parser.add_argument('--tracks_path', default="tracks.json", help='Path to the json file where the tracks will be saved')
    parser.add_argument('--output_path', default="output.mp4", help='Path to the output video file')

    args = parser.parse_args()

    try:
       writer = Writer(args.video_path, args.tracks_path, args.output_path)
       writer.run()

    except Exception as e:
        logger.error(f"Service failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())