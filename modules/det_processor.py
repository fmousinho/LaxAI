import logging
import os
import supervision as sv
import numpy as np
from typing import Optional, Deque, List
from collections import deque, defaultdict
import json

from modules.detection import DetectionModel
from modules.tracker import AffineAwareByteTrack
from modules.utils import log_progress
from config.all_config import detection_config

logger = logging.getLogger(__name__)


class DetectionProcessor:
    """Handles frame-by-frame detection and tracking processing."""
    
    def __init__(self, model: DetectionModel, tracker: AffineAwareByteTrack, detection_file_path: str, nms_iou: Optional[float] = None):
        self.model = model
        self.tracker = tracker
        self.detection_file_path = detection_file_path
        if nms_iou is None:
            nms_iou = detection_config.nms_iou_threshold
        self.nms_iou = nms_iou
        if self.nms_iou is not None:
            logger.info(f"Using NMS IoU threshold: {self.nms_iou}")


    def process_frames(
        self, 
        frames_generator, 
        frame_target: int,
    ) -> List[sv.Detections]:
        """
        Process all frames for detection and tracking.
        
        Args:
            frames_generator: Generator yielding video frames
            frame_target: Total number of frames to process
            detection_file_path: Path to save the detections JSON file
            
        Returns:
            List of sv.Detections objects with loaded detections
        """
        logger.info("Generating detections and tracks for each frame")
        
        multi_frame_detections = list()
        previous_frame = None
        affine_matrix = None
        current_frame_idx = 0
        
        json_sink = sv.JSONSink(self.detection_file_path)
        
        with json_sink as sink:
            for frame in frames_generator:
                if current_frame_idx >= frame_target:
                    break
                    
                # Generate detections for current frame
                frame_detections = self.model.generate_detections(frame)
                
                if len(frame_detections) == 0:
                    # If no detections, create an empty object for len to match frame count
                    frame_detections = self.model.empty_detections()
                else:
           
                    if frame_detections.xyxy.size > 0 and np.any(frame_detections.xyxy < 0):
                        logger.debug(f"Negative coordinates for frame {current_frame_idx}.")

                    if previous_frame is not None:
                        affine_matrix = AffineAwareByteTrack.calculate_affine_transform(previous_frame, frame)
                    if affine_matrix is None:
                        affine_matrix = AffineAwareByteTrack.get_identity_affine_matrix()

                
                frame_detections.metadata.update({'frame_id': current_frame_idx})

                if self.nms_iou is not None and len(frame_detections) > 1:
                    frame_detections = frame_detections.with_nms(threshold=self.nms_iou)

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
                sink.append(detections, custom_data={'frame_id': current_frame_idx})
                
                # Log progress
                log_progress(logger, "Frames read", current_frame_idx, frame_target)
        
        logger.info(f"Detection and tracking processing complete.")
        logger.info(f"JSON saved to: {self.detection_file_path}")
        logger.info(f"Found {self.tracker.get_n_of_tracks()} total tracks.")

        return multi_frame_detections

    def json_to_detections(self, json_file_path: str, video_source: str, update_tracker_state: bool = True) -> List[sv.Detections]:
        """
        Load detections from a JSON file, and optionally updates the tracker internal state.
        
        Args:
            json_file_path: Path to the JSON file containing detections
            update_tracker_state: Whether to update the tracker's internal state with loaded detections
            video_source: Path to the video file

        Returns:
            List of sv.Detections objects with loaded detections
        """
        if not os.path.exists(json_file_path):
            logger.error(f"JSON file not found:")
            logger.error(f"{json_file_path}")
            return []

        logger.info(f"Loading detections from JSON file.")

        rows_by_frame_number = defaultdict(list)
        with open(json_file_path, "r") as f:
            data = json.load(f)
        for row in data:
            frame_number = int(row["frame_id"])
            rows_by_frame_number[frame_number].append(row)

        detections_list = []

        total_frames = sv.VideoInfo.from_video_path(video_path=video_source).total_frames

        for frame_id in range(total_frames):
            if frame_id not in rows_by_frame_number.keys():
                detections = self.model.empty_detections()
            else:
                rows = rows_by_frame_number[frame_id]
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
                        if custom_key in ["x_min", "y_min", "x_max", "y_max", "class_id", "confidence", "tracker_id", "frame_id"]:
                            continue
                        custom_data[custom_key].append(row[custom_key])

                if all([val == "" for val in class_id]):
                    class_id = None
                if all([val == "" for val in confidence]):
                    confidence = None
                if all([val == "" for val in tracker_id]):
                    tracker_id = None

                detections = sv.Detections(
                    xyxy=np.array(xyxy, dtype=np.float32),
                    class_id=np.array(class_id, dtype=int),
                    confidence=np.array(confidence, dtype=np.float32),
                    tracker_id=np.array(tracker_id, dtype=int),
                    data=dict(custom_data))

            detections.metadata.update({'frame_id': frame_id})
            detections_list.append(detections)

        # Update tracker state if requested
        if update_tracker_state and video_source:
            logger.info("Updating tracker state with loaded detections...")
            frames_generator = sv.get_video_frames_generator(source_path=video_source)
            for frame_idx, (frame, detections) in enumerate(zip(frames_generator, detections_list)):
                self.tracker.update_tracks_with_loaded_detections(detections, frame)
                if frame_idx >= len(detections_list) - 1:
                    break
            logger.info("Tracker state updated with loaded detections.")
        
        return detections_list
