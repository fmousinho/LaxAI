import logging
from config.constants import LOGGING_LINE_SIZE
from config.transforms_config import get_transforms, model_config
from modules.utils import log_progress
from typing import Optional, List, Dict, Tuple
import datetime
import os
import torch
import cv2
import supervision as sv
from tqdm import tqdm
from collections import deque
import numpy as np
import tempfile
import shutil

from tools import reporting
from modules.detection import DetectionModel
from modules.player import Player
from tools.store_driver import Store
from modules.tracker import AffineAwareByteTrack, TrackData
from modules.clustering_processor import ClusteringProcessor
from modules.siamesenet import SiameseNet 
from modules.det_processor import DetectionProcessor
from modules.crop_extractor_processor import CropExtractor, create_train_val_split
from modules.dataset import LacrossePlayerDataset
from modules.emb_processor import EmbeddingsProcessor
from modules.writer_processor import VideoWriterProcessor
from modules.player_association import associate_tracks_to_players_with_stitching, stitch_tracks



logger = logging.getLogger(__name__)

_TEAM_COLORS =  {
    0: (255, 0, 0),      # Red for team 0
    1: (0, 0, 255),      # Blue for team 1
    2: (0, 255, 0),      # Green for Referees
    10: (255, 255, 0),   # Yellow for Goalies team 0
    11: (255, 165, 0),   # Orange for Goalies team 1
    -1: (128, 128, 128)  # Gray for unknown
}

_PLAYER_CLASS_ID = 3


def run_application (
        store: Store,
        input_video: str,
        output_video_path: str = "results.mp4",
        device: torch.device = torch.device("cpu"),
        debug_max_frames: Optional[int] = None,
        generate_report: bool = True,
        detections_import_path: Optional[str] = None
    ):

    """
    Main entry point for the video processing application.
    Args:
        store: The Store instance for file management.
        input_video: Path to the input video file.
        output_video_path: Path where the processed video will be saved.
        device: The torch device to use for processing (e.g., "cpu", "cuda").
        debug_max_frames: If set, limits processing to this many frames for debugging.
        detections_import_path: Optional path to save/load detections JSON file. If set, bypass detection step if file exists.
    """

    logger.info("")
    logger.info("run_application called with arguments:")
    logger.info(f"  store:            {type(store).__name__}")
    logger.info(f"  input_video:      {input_video}")
    logger.info(f"  output_video_path:{output_video_path}")
    logger.info(f"  device:           {device}")
    logger.info(f"  debug_max_frames: {debug_max_frames}")
    logger.info(f"  generate_report:  {generate_report}")
    logger.info(f"  detections_import_path: {detections_import_path}")


    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    generator_params = {
        "source_path": input_video,
        "end": debug_max_frames
    }

    # Create a proper temporary directory
    TEMP_DIR = tempfile.mkdtemp(prefix="laxai_", suffix="_temp")
    logger.info(f"Created temporary directory: {TEMP_DIR}")
    
    try:
        # Register cleanup function to remove temp directory on exit
        import atexit
        def cleanup_temp_dir():
            if os.path.exists(TEMP_DIR):
                logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
                shutil.rmtree(TEMP_DIR)
        atexit.register(cleanup_temp_dir)

        FRAME_TARGET:int = debug_max_frames if debug_max_frames is not None else video_info.total_frames #type: ignore

        detection_model = DetectionModel(store=store, device=device)    
        tracker = AffineAwareByteTrack() 

        
        multi_frame_detections = list()



        # --- Generate or load detections and tracks for each frame ---

        frames_generator = sv.get_video_frames_generator(**generator_params)
        detection_save_path = os.path.join(TEMP_DIR, "detections.json")
        detection_processor = DetectionProcessor(detection_model, tracker, detection_save_path)

        if detections_import_path and os.path.exists(detections_import_path):
            logger.info(f"Loading existing detections from {detections_import_path}")
            with open(detections_import_path, 'r') as f:
                multi_frame_detections = detection_processor.json_to_detections(
                    detections_import_path, 
                    update_tracker_state=True, 
                    video_source=input_video
                )
            logger.info(f"Loaded detections for {len(multi_frame_detections)} frames.")
        else:
            if detections_import_path and not os.path.exists(detections_import_path):
                logger.warning(f"Detections import path does not exist: {detections_import_path}")
                logger.warning("Proceeding to generate detections from the video.")
            logger.info("Generating new detections from video.")
            multi_frame_detections = detection_processor.process_frames(
                frames_generator=frames_generator,
                frame_target=FRAME_TARGET
            )


        # --- Extracting crops ---

        frames_generator = sv.get_video_frames_generator(**generator_params)
        crop_processor = CropExtractor(frames_generator, multi_frame_detections, TEMP_DIR)
        crop_processor.extract_crops()


        # --- Merging tracks based on similarity ---

        emb_proc = EmbeddingsProcessor(
            train_dir=crop_processor.get_all_crops_directory(),
            device=device
        )
        emb_proc.setup_model(SiameseNet, inference_only=True)  
        inference_transforms = get_transforms('inference')
        
        def embeddings_processor_with_transforms(crops):
            return emb_proc.create_embeddings_from_crops(
                crops=crops,
                transform=inference_transforms
            )
       
        tracker.create_embeddings_for_tracks(
            embeddings_processor=embeddings_processor_with_transforms
        )

        tracks_data, multi_frame_detections, track_id_mapping = stitch_tracks(
            tracks_data=tracker.get_tracks_data(), 
            multi_frame_detections=multi_frame_detections)
        
        tracker.track_data = tracks_data  
        logger.info("Reorganizing crops based on stitched tracks")
        crop_processor.reorganize_crops_by_stitched_tracks(track_id_mapping)
        logger.info(f"Stitching complete. Now {len(tracks_data)} tracks after stitching.")


        # --- Create train/val split using the standalone function ---

        source_crops_dir = crop_processor.get_all_crops_directory()
        data_dir = crop_processor.get_crops_directory()
        create_train_val_split(source_crops_dir, data_dir)
        
        # --- Train Track Identifier ---

        embeddings_model_path = os.path.join(data_dir, "embeddings_model.pth")
        train_dir = os.path.join(data_dir, "train")
        track_train_processor = EmbeddingsProcessor(train_dir=train_dir, model_save_path=embeddings_model_path, device=device)
        track_train_processor.train_and_save(
            model_class=SiameseNet,
            dataset_class=LacrossePlayerDataset,
            transform=get_transforms('training')
        )

        # --- Cluster Tracks based on their similarity ---

        clustering_processor = ClusteringProcessor(
            model_path=embeddings_model_path,
            all_crops_dir=data_dir,
            clustered_data_dir=os.path.join(data_dir, "clustered"),
            device=device
        )

        num_clusters, num_images = clustering_processor.process_clustering_with_search(
            model_class=SiameseNet,
            source_data_dir=crop_processor.get_all_crops_directory()
        )
        
        logger.info(f"Clustering complete: {num_clusters} clusters from {num_images} images")


        # --- Prepare clustered images for retraining ---

        player_crops_dir = clustering_processor.get_clustered_data_directory()
        data_dir = os.path.join(player_crops_dir, "data")
        create_train_val_split(player_crops_dir, data_dir)


        # --- Train player identifier ---

        embeddings_model_path = os.path.join(data_dir, "player_embeddings_model.pth")
        player_processor = EmbeddingsProcessor(train_dir=os.path.join(data_dir, "train", device=device), model_save_path=embeddings_model_path)
        player_processor.train_and_save(
            model_class=SiameseNet,
            dataset_class=LacrossePlayerDataset,
            transform=get_transforms('training')
        )

        # --- Create embeddings for tracks using the trained model ---

        logger.info("Creating embeddings for tracks using the trained model")
        
        player_emb_proc = EmbeddingsProcessor(
            train_dir=os.path.join(data_dir, "train"), 
            model_save_path=embeddings_model_path
        )
        
        # Load the trained model for inference
        player_emb_proc.setup_model(SiameseNet, inference_only=False)
        
        # Create a wrapper function that uses the EmbeddingsProcessor
        def embeddings_processor_with_trained_model(crops):
            return player_emb_proc.create_embeddings_from_crops(
                crops=crops,
                transform=inference_transforms
            )
        
        # Create embeddings for tracks
        tracker.create_embeddings_for_tracks(
            embeddings_processor=embeddings_processor_with_trained_model,
            device=device
        )

         # --- Analysing Tracks to Create Players  ---

        players: set[Player] = set()  # Set to store unique Player objects
        track_to_player: Dict[int, Player] = {}  # Mapping from tracker ID to Player object
        tracks_data = detection_processor.tracker.get_tracks_data()

        players, track_to_player, multi_frame_detections = associate_tracks_to_players_with_stitching(
            tracks_data, multi_frame_detections
        )

        logger.info(f"Player association complete: {len(players)} unique players from {len(tracks_data)} tracks")

        # --- Write output video using stored detections and validated player info ---

        writer_processor = VideoWriterProcessor(
            output_video_path=output_video_path,
            video_info=video_info,
            team_colors=_TEAM_COLORS
        )
        
        frames_generator = sv.get_video_frames_generator(**generator_params)
        writer_processor.write_annotated_video(
            frames_generator=frames_generator,
            multi_frame_detections=multi_frame_detections,
            track_to_player=track_to_player,
            frame_target=FRAME_TARGET
        )

        # --- (Optional) Fourth pass: Generate analysis report ---
        
        if generate_report:
            logger.info("Generating per-player analysis report")
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_output_dir = os.path.join(reporting.REPORTS_BASE_DIR, run_id)

            # Build a mapping from player_id to tracker_ids and organize by tracks
            player_rows = []
            # Reverse mapping: player_id -> list of tracker_ids
            player_to_trackers = {}
            for tid, player in track_to_player.items():
                if player.id not in player_to_trackers:
                    player_to_trackers[player.id] = []
                player_to_trackers[player.id].append(tid)

            for player in players:
                tracker_ids = player_to_trackers.get(player.id, [])
                
                # Create track-level data for this player
                track_data_list = []
                for tid in tracker_ids:
                    track_data = tracks_data.get(tid, None)
                    if track_data is not None and hasattr(track_data, 'crops'):
                        track_info = {
                            "track_id": tid,
                            "crops": track_data.crops,
                            "frame_first_seen": getattr(track_data, 'frame_first_seen', 'N/A'),
                            "frame_last_seen": getattr(track_data, 'frame_last_seen', 'N/A'),
                            "num_crops": len(track_data.crops),
                            "team": getattr(track_data, 'team', -1)
                        }
                        track_data_list.append(track_info)
                
                player_rows.append({
                    "player_id": player.id,
                    "num_tracks": len(tracker_ids),
                    "total_crops": sum(track["num_crops"] for track in track_data_list),
                    "tracks": track_data_list  # List of track dictionaries with their crops
                })

            logger.info(f"Found {len(player_rows)} players to report on.")
            reporting.generate_player_report_html(run_id, run_output_dir, player_rows)
            reporting.update_main_index_html()
    
    finally:
        # Ensure cleanup happens even if an error occurs
        if os.path.exists(TEMP_DIR):
            logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)
