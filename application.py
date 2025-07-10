import logging
from config.constants import LOGGING_LINE_SIZE
from config.transforms_config import get_transforms
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

from tools import reporting
from modules.detection import DetectionModel
from modules.player import Player
from modules.team_identification import TeamIdentifier, PlayerMasker
from tools.store_driver import Store
from modules.tracker import AffineAwareByteTrack, TrackData
from modules.clustering_processor import ClusteringProcessor
from modules.siamesenet import SiameseNet  # Your model class
from modules.detection_processor import DetectionProcessor
from modules.crop_extractor_processor import CropExtractor, create_train_val_split
from modules.dataset import LacrossePlayerDataset, data_transforms
from modules.train_processor import Trainer
from modules.writer_processor import VideoWriterProcessor
from modules.player_association import (
    associate_tracks_to_players_greedy,
    associate_tracks_to_players_globally
)


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
_N_FRAMES_FOR_TEAM_ID = 100 


def run_application (
        store: Store,
        input_video: str,
        output_video_path: str = "results.mp4",
        device: torch.device = torch.device("cpu"),
        debug_max_frames: Optional[int] = None,
        generate_report: bool = True
    ):

    """
    Main entry point for the video processing application.
    Args:
        store: The Store instance for file management.
        input_video: Path to the input video file.
        output_video_path: Path where the processed video will be saved.
        device: The torch device to use for processing (e.g., "cpu", "cuda").
        debug_max_frames: If set, limits processing to this many frames for debugging.
    """

    logger.info("")
    logger.info("run_application called with arguments:")
    logger.info(f"  store:            {type(store).__name__}")
    logger.info(f"  input_video:      {input_video}")
    logger.info(f"  output_video_path:{output_video_path}")
    logger.info(f"  device:           {device}")
    logger.info(f"  debug_max_frames: {debug_max_frames}")
    logger.info(f"  generate_report:  {generate_report}")


    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    generator_params = {
        "source_path": input_video,
        "end": debug_max_frames
    }

    TEMP_DIR = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)

    FRAME_TARGET = debug_max_frames if debug_max_frames is not None else video_info.total_frames

    detection_model = DetectionModel(store=store, device=device)    
    tracker = AffineAwareByteTrack() 
    team_identifier = TeamIdentifier()
    masker = PlayerMasker()
    
    multi_frame_detections = list()



    # --- Generate detections and tracks for each frame ---
    
    frames_generator = sv.get_video_frames_generator(**generator_params)
    detection_file_path = os.path.join(TEMP_DIR, "detections.json")
    detection_processor = DetectionProcessor(detection_model, tracker, detection_file_path)
    
    multi_frame_detections = detection_processor.process_frames(
        frames_generator=frames_generator,
        frame_target=FRAME_TARGET
    )

    # --- Extracting crops and creating training/validation splits ---

    frames_generator = sv.get_video_frames_generator(**generator_params)
    # Create a copy of detections for crop extraction
    crop_processor = CropExtractor(frames_generator, multi_frame_detections, TEMP_DIR)

    crop_processor.extract_crops()
    
    # Create train/val split using the standalone function
    source_crops_dir = crop_processor.get_all_crops_directory()
    data_dir = crop_processor.get_crops_directory()
    create_train_val_split(source_crops_dir, data_dir)
    
    # --- Train Track Identifier ---

    embeddings_model_path = os.path.join(data_dir, "embeddings_model.pth")
    train_dir = os.path.join(data_dir, "train")
    track_train_processor = Trainer(train_dir=train_dir, model_save_path=embeddings_model_path)
    track_train_processor.train_and_save(
        model_class=SiameseNet,
        dataset_class=LacrossePlayerDataset,
        transform=data_transforms
    )

    # --- Cluster Tracks based on their similarity ---

    clustering_processor = ClusteringProcessor(
        model_path=embeddings_model_path,
        all_crops_dir=data_dir,
        clustered_data_dir=os.path.join(data_dir, "clustered"),
        embedding_dim=128,
        batch_size=32,
        dbscan_min_samples=5
    )

    num_clusters, num_images = clustering_processor.process_clustering_with_search(
        model_class=SiameseNet,
        source_data_dir=crop_processor.get_all_crops_directory(),
        target_min_clusters=20,
        target_max_clusters=40,
        initial_eps=0.6,
        device=device
    )
    

    # --- Prepare clustered images for retraining ---

    player_crops_dir = clustering_processor.get_clustered_data_directory()
    data_dir = os.path.join(player_crops_dir, "data")
    create_train_val_split(player_crops_dir, data_dir)

    # --- Train player identifier ---

    embeddings_model_path = os.path.join(data_dir, "player_embeddings_model.pth")
    player_processor = Trainer(train_dir=os.path.join(data_dir, "train"), model_save_path=embeddings_model_path)
    player_processor.train_and_save(
        model_class=SiameseNet,
        dataset_class=LacrossePlayerDataset,
        transform=data_transforms
    )

    # --- Create embeddings for tracks using the trained model ---

    logger.info("Creating embeddings for tracks using the trained model")
    
    # Load the trained model
    trained_model = SiameseNet(embedding_dim=128)
    trained_model.load_state_dict(torch.load(embeddings_model_path, map_location=device))
    trained_model.to(device)
    trained_model.eval()
    
    # Create a simple embeddings processor function
    def embeddings_processor(crops_tensor):
        """Simple wrapper for the model forward pass."""
        return trained_model(crops_tensor)
    
    # Create embeddings for tracks
    tracker.create_embeddings_for_tracks(
        embeddings_processor=embeddings_processor,
        device=device
    )

    tracker_data = detection_processor.tracker.get_tracks_data()
    tracker_data = {tid: data for tid, data in tracker_data.items() if data.class_id == _PLAYER_CLASS_ID}



     # --- Analysing Tracks to Create Players  ---

    players: set[Player] = set()  # Set to store unique Player objects
    track_to_player: Dict[int, Player] = {}  # Mapping from tracker ID to Player object
    tracks_data = detection_processor.tracker.get_tracks_data()
    
    # Association parameters
    SIMILARITY_THRESHOLD = 0.9  # Minimum cosine similarity for association

    players, track_to_player = associate_tracks_to_players_globally(tracks_data, SIMILARITY_THRESHOLD)

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
    generate_report = True
    if generate_report:
        logger.info("Generating per-track analysis report")
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(reporting.REPORTS_BASE_DIR, run_id)

        # Collect per-track info: track_id, team, player_id
        per_track_rows = []
        for tid, track_data in tracker_data.items():
            player = track_to_player.get(tid, None)
            per_track_rows.append({
                "track_id": tid,
                "original_crop": getattr(track_data, '_crops', [None])[0] if hasattr(track_data, '_crops') and track_data._crops else None,
                "masked_crop": None,  # Masked crops not available in current workflow
                "team": getattr(track_data, 'team', -1),
                "player_id": player.id if player is not None else -1
            })

        logger.info(f"Found {len(per_track_rows)} tracks to report on.")
        reporting.generate_track_report_html(run_id, run_output_dir, per_track_rows)
        reporting.update_main_index_html()
