import logging
import os
import supervision as sv
import numpy as np
from typing import Optional, Deque
from collections import deque, defaultdict
import json

from modules.detection import DetectionModel
from modules.custom_tracker import AffineAwareByteTrack
from modules.utils import log_progress

logger = logging.getLogger(__name__)

# Progress logging configuration
PROGRESS_LOG_PERCENTAGE = 5  # Log every 5% progress


class DetectionProcessor:
    """Handles frame-by-frame detection and tracking processing."""
    
    def __init__(self, model: DetectionModel, tracker: AffineAwareByteTrack, detection_file_path: str):
        self.model = model
        self.tracker = tracker
        self.detection_file_path = detection_file_path
        
    def process_frames(
        self, 
        frames_generator, 
        frame_target: int
    ) -> Deque[sv.Detections]:
        """
        Process all frames for detection and tracking.
        
        Args:
            frames_generator: Generator yielding video frames
            frame_target: Total number of frames to process
            
        Returns:
            Deque of detections for each frame
        """
        logger.info("Generating detections and tracks for each frame")
        
        multi_frame_detections = deque()
        previous_frame = None
        affine_matrix = None
        current_frame_idx = 0
        
        # Calculate progress step for logging
        progress_step = max(1, frame_target * PROGRESS_LOG_PERCENTAGE // 100)
        
        json_sink = sv.JSONSink(self.detection_file_path)
        
        with json_sink as sink:
            for frame in frames_generator:
                if current_frame_idx >= frame_target:
                    break
                    
                # Generate detections for current frame
                frame_detections = self.model.generate_detections(frame)
                
                # Validate detections
                if frame_detections.xyxy.size > 0 and np.any(frame_detections.xyxy < 0):
                    logger.warning("Detections contain negative xyxy coordinates. May be an issue (or not).")

                # Calculate affine transformation between frames
                if previous_frame is not None:
                    affine_matrix = AffineAwareByteTrack.calculate_affine_transform(previous_frame, frame)
                if affine_matrix is None:
                    affine_matrix = AffineAwareByteTrack.get_identity_affine_matrix()

                # Update tracker with detections and affine transformation
                detections = self.tracker.update_with_transform(
                    detections=frame_detections,
                    frame=frame,
                    affine_matrix=affine_matrix
                )

                # Store detections and update state
                multi_frame_detections.append(detections)
                previous_frame = frame.copy()
                current_frame_idx += 1
                
                # Write detections to JSON file
                sink.append(detections, custom_data={'frame_index': current_frame_idx})
                
                # Log progress
                log_progress(logger, "Frames read", current_frame_idx, frame_target, progress_step)
        
        logger.info(f"Detection processing complete. JSON saved to: {self.detection_file_path}")
        return multi_frame_detections

    def json_to_detections(json_file_path: str) -> List[sv.Detections]:
        """
        Load detections from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing detections

        Returns:
            List of sv.Detections objects with loaded detections
        """
        if not os.path.exists(json_file_path):
            logger.error(f"JSON file not found:")
            logger.error(f"{json_file_path}")
            return []

        logger.info(f"Loading detections from JSON file.")

        rows_by_frame_number = defaultdict(list)
        with open(json_file, "r") as f:
            data = json.load(f)
        for row in data:
            frame_number = int(row["frame_id"])
            rows_by_frame_number[frame_number].append(row)

        detections_list = []
        for frame_number, rows in rows_by_frame_number.items():
            xyxy = []
            class_id = []
            confidence = []
            tracker_id = []
            custom_data = defaultdict(list)

            for row in rows:
                xyxy.append([row[key] for key in ["x_min", "y_min", "x_max", "y_max"]])
                class_id.append(row["class_id"])
                confidence.append(row["confidence"])
                tracker_id.append(row["tracker_id"])

                for custom_key in row.keys():
                    if custom_key in ["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id"]:
                        continue
                    custom_data[custom_key].append(row[custom_key])

            if all([val == "" for val in class_id]):
                class_id = None
            if all([val == "" for val in confidence]):
                confidence = None
            if all([val == "" for val in tracker_id]):
                tracker_id = None

            detections_list.append(
                sv.Detections(
                    xyxy=np.array(xyxy, dtype=np.float32),
                    class_id=np.array(class_id, dtype=int),
                    confidence=np.array(confidence, dtype=np.float32),
                    tracker_id=np.array(tracker_id, dtype=int),
                    data=dict(custom_data)
                )
            )
        return detections_list
