import logging
logger = logging.getLogger(__name__)

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import tempfile
import urllib.request
import os

import cv2
import json
import numpy as np

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.utils.id_generator import create_simple_uuid
from schemas.tracking import TrackingParams
from shared_libs.common.detection import DetectionModel

from detection import DetectionModel
from tracker.byte_tracker import ByteTracker
from tracker.cam_mvmt import calculate_transform

DEFAULT_CONFIDENCE_THRESHOLD = 0.4

class TrackingController:
    """
    Orchestrates the end-to-end unverified track generation process.
    Handles video discovery, pipeline execution, and status reporting.
    """

    def __init__(
        self,
        tracking_params: TrackingParams,
        wandb_run_name: str = "track_generation_run",
    ):
        self.tracking_params = tracking_params
        self.wandb_run_name = wandb_run_name

        self.storage_client = get_storage(tenant_id)
        self.path_manager = GCSPaths()

        self.detector = DetectionModel()
        self.tracker = ByteTracker(self.tracking_params)

    def run (self, video_path: str, tracks_save_path: str):
        """
        Generates detections and tracks from a given video.

        Args:
            video_path: Path to the video file (URL or local path).

        Returns:
            TBD.
        """
        logger.info(f"Starting Tracking Controller for tenant: {self.tenant_id}")

        original_video_path = video_path
        # Gets video into cv2
        if video_path.startswith(('http://', 'https://')):
            logger.info(f"Downloading video from URL: {video_path}")
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_file:
                urllib.request.urlretrieve(video_path, temp_file.name)
                video_path = temp_file.name
                cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")

        frame_count = 0
        prev_frame = None
        all_tracks = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            detections_obj = self.detector.predict(frame, threshold=DEFAULT_CONFIDENCE_THRESHOLD)
            # Detections need to be properly formatted for ByteTracker
            # ByteTracker expects: [x1, y1, x2, y2, score]
            # detection_obj.xyxy is [N, 4], detection_obj.confidence is [N]

            # Filter out non-player detections
            player_mask = detections_obj.class_ids == 3
            detections_obj = detections_obj[player_mask]
            
            if detections_obj.xyxy.shape[0] > 0:
                 detections = np.column_stack((detections_obj.xyxy, detections_obj.confidence))
            else:
                detections = np.empty((0, 5))

            if frame_count > 0:
                S, T = calculate_transform(prev_frame, frame)
            else:
                S, T = None, None
            
            frame_tracks = self.tracker.update(detections, S, T)
            all_tracks.append(frame_tracks)
            frame_count += 1
            prev_frame = frame

        cap.release()

        self.save_tracks(all_tracks, original_video_path, tracks_save_path)
        return True

    def save_tracks(self, tracks_list: List[List], video_source: str, tracks_save_path: str):
        serialized_tracks = []
        for frame_tracks in tracks_list:
            for track in frame_tracks:
                # Convert STrack object to dictionary
                track_dict = {
                    "frame_id": int(track.frame_id),
                    "track_id": int(track.track_id),
                    "bbox": [float(x) for x in track.tlwh],  # tlwh format
                    "score": float(track.score)
                }
                serialized_tracks.append(track_dict)
        
        output_data = {
            "video_source": video_source,
            "tracks": serialized_tracks
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(tracks_save_path)), exist_ok=True)

        with open(tracks_save_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        




        




