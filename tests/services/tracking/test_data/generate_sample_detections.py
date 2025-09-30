#!/usr/bin/env python3
"""
Script to generate detections.json file from the test video.

This script processes the test video file and generates real detection data
that can be used for testing detection-related functionality.
"""

import sys
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import cv2
import numpy as np
from shared_libs.common.detection_utils import detections_to_json
from shared_libs.config.all_config import detection_config
from shared_libs.common.detection import DetectionModel


def process_video_for_detections(video_path: str, max_frames: int = 50) -> list:
    """Process video frames and generate detections.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to process

    Returns:
        List of Detections objects, one per processed frame
    """
    # Initialize detection model
    detection_model = DetectionModel()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    all_detections = []
    frame_count = 0

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB (if needed by the model)
            if frame.shape[-1] == 3:  # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            try:
                # Generate detections for this frame
                frame_detections = detection_model.generate_detections(frame_rgb)

                if frame_detections is not None and len(frame_detections) > 0:
                    # Add frame index to the detections data
                    if hasattr(frame_detections, 'data') and frame_detections.data:
                        frame_detections.data['frame_index'] = [frame_count] * len(frame_detections)
                    else:
                        frame_detections.data = {'frame_index': [frame_count] * len(frame_detections)}

                    # Set metadata
                    frame_detections.metadata = {
                        'frame_id': frame_count,
                        'video_path': video_path,
                        'resolution': f"{frame.shape[1]}x{frame.shape[0]}"
                    }

                    all_detections.append(frame_detections)
                    print(f"Frame {frame_count}: {len(frame_detections)} detections")
                else:
                    print(f"Frame {frame_count}: No detections")

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue

            frame_count += 1

    finally:
        cap.release()

    print(f"Processed {frame_count} frames, generated detections for {len(all_detections)} frames")
    return all_detections


def save_detections_to_json(detections_list: list, output_path: str):
    """Save a list of Detections objects to a JSON file.

    Args:
        detections_list: List of Detections objects
        output_path: Path to save the JSON file
    """
    from shared_libs.common.detection_utils import detection_to_json_single

    # Convert each Detections object to JSON format
    json_data = []
    for detections in detections_list:
        # Convert the Detections object directly to a single dict
        frame_json = detection_to_json_single(detections)
        json_data.append(frame_json)

    # Save to file
    with open(output_path, 'w') as f:
        json.dump({"detections": json_data}, f, indent=2)

    print(f"Saved {len(json_data)} detections from {len(detections_list)} frames to {output_path}")


def main():
    """Main function to generate detections.json from test video."""
    # Path to the test video (in the same directory as this script)
    video_path = "test_video.mp4"

    if not Path(video_path).exists():
        print(f"Error: Test video not found at {video_path}")
        return

    print(f"Processing video: {video_path}")

    # Process video and generate detections
    try:
        detections_list = process_video_for_detections(video_path, max_frames=30)  # Process first 30 frames

        if not detections_list:
            print("No detections generated from video")
            return

        # Save to detections.json in the same directory as the video
        output_path = "detections.json"
        save_detections_to_json(detections_list, output_path)

        # Also save to the root data directory for general testing
        root_output_path = "../../../data/detections.json"
        save_detections_to_json(detections_list, root_output_path)

    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()