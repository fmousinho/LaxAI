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

from detection import DetectionModel
from tracker.byte_tracker import BYTETracker
from tracker.cam_mvmt import calculate_transform

DEFAULT_CONFIDENCE_THRESHOLD = 0.4

class TrackingController:
    """
    Orchestrates the end-to-end unverified track generation process.
    Handles video discovery, pipeline execution, and status reporting.
    """

    def __init__(self, tracking_params: TrackingParams, wandb_run_name: Optional[str] = None, wandb_config: Optional[Dict] = None):
        self.tracking_params = tracking_params
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config

        #self.storage_client = get_storage(tenant_id)
        #self.path_manager = GCSPaths()

        self.detector = DetectionModel()
        self.tracker = BYTETracker(self.tracking_params)

    def run (self, video_path: str, tracks_save_path: str, detections_save_path: Optional[str] = None):
        """
        Generates detections and tracks from a given video.

        Args:
            video_path: Path to the video file (URL or local path).
            tracks_save_path: Path to the json file where the tracks will be saved.
            detections_save_path: Path to the json file where the detections will be saved.

        Returns:
            TBD.
        """
        logger.info(f"Starting Tracking Controller")

        original_video_path = video_path
        # Gets video into cv2
        # If it's a GCS path, download it locally
        if video_path.startswith("gs://"):
            local_path = f"/tmp/{os.path.basename(video_path)}"
            download_blob(video_path, local_path)
            video_path = local_path
        elif video_path.startswith(('http://', 'https://')):
            logger.info(f"Downloading video from URL: {video_path}")
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_file:
                urllib.request.urlretrieve(video_path, temp_file.name)
                video_path = temp_file.name
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file {video_path}")
            return False

        frame_count = 0
        prev_frame = None
        all_tracks = []
        all_detections = []
        ttl_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            detections_obj = self.detector.predict(frame, threshold=DEFAULT_CONFIDENCE_THRESHOLD)

            # Filter out non-player detections
            player_mask = detections_obj.class_id == 3
            detections_obj = detections_obj[player_mask]
            
            if len(detections_obj.xyxy) > 0:
                 detections = np.column_stack((detections_obj.xyxy, detections_obj.confidence))
            else:
                detections = np.empty((0, 5))

            if frame_count > 0:
                S, T = calculate_transform(prev_frame, frame)
            else:
                S, T = None, None
            
            frame_tracks = self.tracker.assign_tracks_to_detections(detections, S, T)
            all_tracks.append(frame_tracks.copy())
            if detections_save_path is not None:
                all_detections.append(detections.copy())
            
            frame_count += 1
            prev_frame = frame
            
            # Log progress
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count}/{ttl_frames} frames")

        cap.release()

        # Save tracks using unified serialization module
        track_serialization.save(
            objects_per_frame=all_tracks,
            video_source=original_video_path,
            save_path=tracks_save_path,
            use_track_id=True
        )
        
        if detections_save_path is not None:
            track_serialization.save(
                objects_per_frame=all_detections,
                video_source=original_video_path,
                save_path=detections_save_path,
                use_track_id=False
            )
            
        return True


