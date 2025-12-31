import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch

from shared_libs.common.wandb_logger import WandbLogger
from shared_libs.common import track_serialization
from shared_libs.common.model import ReIdModel

from tracker.config import TrackingParams
from tracker.detection import DetectionModel
from tracker.tracker import Tracker
from tracker.cam_mvmt import calculate_transform
from tracking_utils import emb_files

logger = logging.getLogger(__name__)

FRAMES_PER_LOG_MSG = 50
PLAYER_CLASS_ID = 3  
''' Classes used during RFDETR training:
    1: goalkeeper
    2: not_player (not used)
    3: player
    4: player_opposing_team (not used)
    5: referee
'''

class TrackingController:
    """
    Orchestrates the end-to-end unverified track generation process.
    Handles video discovery, pipeline execution, and status reporting.
    """

    def __init__(
        self,
        tracking_params: TrackingParams,
    ):
        self.params = tracking_params
        self.detector = DetectionModel()
        self.device_str = "cpu"
        self.reid_model = None

        if self.params.enable_reid:
            self.wandb_logger = WandbLogger(api_key=self.params.wandb_api_key)
            self.reid_model = self.wandb_logger.load_model_from_registry(
                model_class=ReIdModel,
                collection_name=ReIdModel.model_name,
                alias="latest",
                device=self.device_str,
                pretrained=False,
            )
            if self.reid_model is None:
                raise ValueError("ReID model not found in registry")
            self.reid_model.eval()
        
        self.tracker = Tracker(
            self.params,
            reid_model=self.reid_model,
        )
            

    def run(
        self,
        video_path: str,
        tracks_save_path: str,
        detections_save_path: Optional[str] = None,
        embeddings_save_path: Optional[str] = None,
    ):
        """
        Generates detections and tracks from a given video.

        Args:
            video_path: Path to the video file (URL or local path).
            tracks_save_path: Path to the json file where the tracks will be
                saved.
            detections_save_path: Path to the json file where the detections
                will be saved.

        Returns:
            TBD.
        """
        logger.info("Starting Tracking Controller")

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

            detections_obj = self.detector.predict(
                frame,
                threshold=self.params.default_confidence_threshold,
            )

            # Filter out non-player detections
            player_mask = detections_obj.class_id == PLAYER_CLASS_ID
            detections_obj = detections_obj[player_mask]
            
            if len(detections_obj.xyxy) > 0:
                 detections = np.column_stack(
                     (detections_obj.xyxy, detections_obj.confidence)
                 )
                 
                 # Apply detection filtering pipeline (NMS + border removal)
                 from tracking_utils.detection_filters import (
                     filter_detections_pipeline,
                 )
                 detections = filter_detections_pipeline(
                     detections,
                     frame_size=(frame.shape[1], frame.shape[0]),
                     # (width, height)
                     track_predictions=None,  # Will be handled in byte_tracker
                     nms_iou_threshold=
                     self.params.nms_iou_threshold,
                     border_margin=
                     self.params.border_margin,
                 )
            else:
                detections = np.empty((0, 5))

            if frame_count > 0 and prev_frame is not None:
                S, T = calculate_transform(prev_frame, frame)
            else:
                S, T = None, None
            
            frame_tracks = self.tracker.assign_tracks_to_detections(
                detections,
                S,
                T,
                frame,
            )
            
            all_tracks.append(frame_tracks.copy())
            if detections_save_path is not None:
                all_detections.append(detections.copy())
            
            frame_count += 1
            prev_frame = frame
            
            # Log progress
            if frame_count % FRAMES_PER_LOG_MSG == 0:
                logger.info(f"Processed {frame_count}/{ttl_frames} frames")

        cap.release()
        
        # Log prediction accuracy statistics
        logger.info(
            "Video processing complete. Logging prediction statistics..."
        )
        self.tracker.log_prediction_statistics()

        # Save tracks using unified serialization module
        track_serialization.save(
            objects_per_frame=all_tracks,
            video_source=video_path,
            save_path=tracks_save_path
        )
        
        if detections_save_path is not None:
            track_serialization.save(
                objects_per_frame=all_detections,
                video_source=video_path,
                save_path=detections_save_path
        )

        if embeddings_save_path is not None:
            all_stracks = {}

            ttl_embeddings = 0

            tracks_types_to_save = [
                self.tracker.tracked_stracks,
                self.tracker.lost_stracks,
                self.tracker.removed_stracks,
            ]

            for tracks in tracks_types_to_save:
                for track in tracks:
                    features_list = track.features
                    if not features_list:
                        continue
                    
                    # Stack them: List[(1, D)] -> (N, D)
                    # Assumes features are (1, D) tensors
                    stacked_feats = torch.cat(features_list, dim=0).cpu()
                    
                    # Calculate mean for backward compatibility
                    mean_feat = torch.mean(stacked_feats, dim=0)
                    variance_feat = torch.var(stacked_feats, dim=0)
                    
                    all_stracks[track.track_id] = {
                        'mean': mean_feat,
                        'variance': variance_feat,
                        'all': stacked_feats,
                        'count': track.features_count
                    }
                    ttl_embeddings += track.features_count
        
            
            logger.info(
                f"Saving {len(all_stracks)} tracks "
                f"({ttl_embeddings} total embeddings) "
                f"to {embeddings_save_path}"
            )
            emb_files.save_embeddings(all_stracks, embeddings_save_path)
            
        return True


