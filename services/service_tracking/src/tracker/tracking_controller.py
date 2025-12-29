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
import torch

from shared_libs.common.wandb_logger import wandb_logger
from shared_libs.common import track_serialization
from shared_libs.common.model import ReIdModel

from tracker.config import TrackingParams
from tracker.detection import DetectionModel
from tracker.byte_tracker import BYTETracker
from tracker.cam_mvmt import calculate_transform
from tracking_utils import emb_files

ENABLE_REID = True
DEFAULT_CONFIDENCE_THRESHOLD = 0.4

class TrackingController:
    """
    Orchestrates the end-to-end unverified track generation process.
    Handles video discovery, pipeline execution, and status reporting.
    """

    def __init__(self, tracking_params: TrackingParams, wandb_run_name: Optional[str] = None):
        self.tracking_params = tracking_params
        self.wandb_run_name = wandb_run_name
        self.detector = DetectionModel()
        self.device_str = "cpu"
        self.reid_model = None

        if ENABLE_REID:
            self.wandb_logger = wandb_logger
            self.reid_model = self.wandb_logger.load_model_from_registry(
                        model_class=ReIdModel,
                        collection_name=ReIdModel.model_name,
                    alias="latest",
                    device=self.device_str,
                    pretrained=False
                )
            if self.reid_model is None:
                raise ValueError("ReID model not found in registry")
            self.reid_model.eval()
        
        self.tracker = BYTETracker(self.tracking_params, reid_model=self.reid_model)
            

    def run (self, video_path: str, tracks_save_path: str, detections_save_path: Optional[str] = None, embeddings_save_path: Optional[str] = None):
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

        # Verify the video path is a local file
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
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
                 
                 # Apply detection filtering pipeline (NMS + border removal)
                 from tracking_utils.detection_filters import filter_detections_pipeline
                 detections = filter_detections_pipeline(
                     detections,
                     frame_size=(frame.shape[1], frame.shape[0]),  # (width, height)
                     track_predictions=None,  # Will be handled in byte_tracker
                     nms_iou_threshold=0.3,
                     border_margin=2,
                 )
            else:
                detections = np.empty((0, 5))

            if frame_count > 0 and prev_frame is not None:
                S, T = calculate_transform(prev_frame, frame)
            else:
                S, T = None, None
            
            frame_tracks = self.tracker.assign_tracks_to_detections(detections, S, T, frame)
            
            all_tracks.append(frame_tracks.copy())
            if detections_save_path is not None:
                all_detections.append(detections.copy())
            
            frame_count += 1
            prev_frame = frame
            
            # Log progress
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count}/{ttl_frames} frames")

        cap.release()
        
        # Log prediction accuracy statistics
        logger.info(f"Video processing complete. Logging prediction statistics...")
        self.tracker.log_prediction_statistics()

        # Save tracks using unified serialization module
        track_serialization.save(
            objects_per_frame=all_tracks,
            video_source=original_video_path,
            save_path=tracks_save_path
        )
        
        if detections_save_path is not None:
            track_serialization.save(
                objects_per_frame=all_detections,
                video_source=original_video_path,
                save_path=detections_save_path
        )

        if embeddings_save_path is not None:
            # Collect all tracks (deduplicated by track_id)
            # The tracker keeps track of tracked, lost, and removed.
            # Using byte_tracker.joint_stracks to get unique objects might be useful, or manual dict.
            all_stracks = {}
            
            # Helper to add tracks
            def add_unique_tracks(tracks):
                for t in tracks:
                    if t.track_id not in all_stracks and t.features is not None:
                        all_stracks[t.track_id] = t
            
            add_unique_tracks(self.tracker.tracked_stracks)
            add_unique_tracks(self.tracker.lost_stracks)
            add_unique_tracks(self.tracker.removed_stracks)
            
            # Format for saving
            embeddings_dict = {}
            for tid, t in all_stracks.items():
                features_list = t.features
                if not features_list:
                    continue
                    
                # Stack them: List[(1, D)] -> (N, D)
                # Assumes features are (1, D) tensors
                stacked_feats = torch.cat(features_list, dim=0).cpu()
                
                # Calculate mean for backward compatibility
                mean_feat = torch.mean(stacked_feats, dim=0)
                
                embeddings_dict[tid] = {
                    'mean': mean_feat,
                    'all': stacked_feats,
                    'count': t.features_count
                }
            
            logger.info(f"Saving {len(embeddings_dict)} track embeddings to {embeddings_save_path}")
            emb_files.save_embeddings(embeddings_dict, embeddings_save_path)
            
        return True


