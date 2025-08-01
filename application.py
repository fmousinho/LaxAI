import logging
from src.config.all_config import (
    debug_config,
    detection_config,
    model_config,
    track_stitching_config,
    transform_config,
)
from src.config.transforms import (
    get_transforms,
    initialize_background_removal,
    refresh_transform_instances,
)
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
from src.common import (
    DetectionModel,
    create_train_val_split,
    extract_crops_from_video,
    load_detections_from_json,
    process_frames,
    reorganize_crops_by_stitched_tracks,
    get_storage,
)
from modules.player import Player
from modules.tracker import AffineAwareByteTrack, TrackData
from modules.clustering_processor import ClusteringProcessor
from src.train import LacrossePlayerDataset, SiameseNet
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

def run_application (
        tenant_id: str,
        input_video: str,
        output_video_path: str = detection_config.output_video_path,
        device: torch.device = torch.device("cpu"),
        debug_max_frames: Optional[int] = None,
        generate_report: bool = True,
        detections_import_path: Optional[str] = None,
        temp_dir: Optional[str] = None
    ):

    """
    Main entry point for the video processing application.
    Args:
        tenant_id: The tenant ID for GCS operations.
        input_video: Path to the input video file.
        output_video_path: Path where the processed video will be saved.
        device: The torch device to use for processing (e.g., "cpu", "cuda").
        debug_max_frames: If set, limits processing to this many frames for debugging.
        detections_import_path: Optional path to save/load detections JSON file. If set, bypass detection step if file exists.
    """

    logger.info("")
    logger.info("run_application called with arguments:")
    logger.info(f"  tenant_id:        {tenant_id}")
    logger.info(f"  input_video:      {input_video}")
    logger.info(f"  output_video_path:{output_video_path}")
    logger.info(f"  device:           {device}")
    logger.info(f"  debug_max_frames: {debug_max_frames}")
    logger.info(f"  generate_report:  {generate_report}")
    logger.info(f"  detections_import_path: {detections_import_path}")
    logger.info(f"  temp_dir:         {temp_dir}")


    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    generator_params = {
        "source_path": input_video,
        "end": debug_max_frames
    }
    if temp_dir is None:
        raise ValueError("temp_dir must be provided by the caller.")
    TEMP_DIR = temp_dir
    logger.info(f"Using temporary directory: {TEMP_DIR}")

    FRAME_TARGET:int = debug_max_frames if debug_max_frames is not None else video_info.total_frames #type: ignore

    detection_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #MPS workaround
    admin_storage = get_storage("common")
    detection_model = DetectionModel(store=admin_storage, device=detection_device)
    tracker = AffineAwareByteTrack() 


    # --- Generate or load detections and tracks for each frame ---
    frames_generator = sv.get_video_frames_generator(**generator_params)
    if debug_config.save_detections_file:
        detection_save_path = debug_config.save_detections_file
    else:
        detection_save_path = os.path.join(TEMP_DIR, "detections.json")

    if detections_import_path and os.path.exists(detections_import_path):
        logger.info(f"Loading existing detections from {detections_import_path}")
        multi_frame_detections = load_detections_from_json(
            detection_model,
            tracker,
            detections_import_path,
            input_video,
            update_tracker_state=True
        )
        logger.info(f"Loaded detections for {len(multi_frame_detections)} frames.")
    else:
        if detections_import_path and not os.path.exists(detections_import_path):
            logger.warning(f"Detections import path does not exist: {detections_import_path}")
            logger.warning("Proceeding to generate detections from the video.")
        logger.info("Generating new detections from video.")
        multi_frame_detections = process_frames(
            detection_model,
            tracker,
            frames_generator,
            FRAME_TARGET,
            detection_save_path
        )


        # --- Extracting crops ---

        frames_generator = sv.get_video_frames_generator(**generator_params)
        crops_dir, all_crops_dir = extract_crops_from_video(frames_generator, multi_frame_detections, TEMP_DIR)

        # --- Setup inference transforms (used throughout the pipeline) ---
        transform_config.enable_background_removal = True
        stride = video_info.total_frames // transform_config.background_detector_sample_frames
        grass_crop_generator = sv.get_video_frames_generator(**generator_params, stride=stride )        
        sample_images = [frame for frame in grass_crop_generator]
        initialize_background_removal(sample_images)
        refresh_transform_instances()

        inference_transforms = get_transforms('opencv_safe')

        # --- Merging tracks based on similarity ---

        if track_stitching_config.stich_tracks_after_tracker == True:
            logger.info("Stiching tracks based on similarity")

            # Create embeddings for tracks before stitching
            create_track_embeddings(
                tracker=tracker,
                train_dir=all_crops_dir,
                device=device,
                inference_only=True
            )

            tracks_data, multi_frame_detections, track_id_mapping = stitch_tracks(
                tracks_data=tracker.get_tracks_data(), 
                multi_frame_detections=multi_frame_detections)
            
            tracker.track_data = tracks_data  
            logger.info("Reorganizing crops based on stitched tracks")
            reorganize_crops_by_stitched_tracks(all_crops_dir, TEMP_DIR, track_id_mapping)
            logger.info(f"Stitching complete. Now {len(tracks_data)} tracks after stitching.")


        # --- Create train/val split using the standalone function ---

        source_crops_dir = all_crops_dir
        data_dir = crops_dir
        create_train_val_split(source_crops_dir, data_dir)
        
        # --- Analysing Tracks to Create Players  ---

        players: set[Player] = set()  # Set to store unique Player objects
        track_to_player: Dict[int, Player] = {}  # Mapping from tracker ID to Player object
        tracks_data = tracker.get_tracks_data()

        if not debug_config.bypass_player_creation:
            players, track_to_player, multi_frame_detections = run_player_training_pipeline(
                data_dir=data_dir,
                all_crops_dir=all_crops_dir,
                temp_dir=temp_dir,
                device=device,
                tracker=tracker,
                tracks_data=tracks_data,
                multi_frame_detections=multi_frame_detections
            )
        else:
            logger.info("Bypassing player creation as per debug configuration.")
            for track_id, track_data in tracks_data.items():
                placeholder_player = Player(
                    tracker_id=track_id,
                    initial_embedding=None,
                    initial_crops=track_data.crops if track_data else [],
                    team=-1  # Unknown team
                )
                players.add(placeholder_player)
                track_to_player[track_id] = placeholder_player


        
        # --- Ensure all tracker IDs in detections have player mappings ---
        
        players, track_to_player = ensure_player_mappings(
            multi_frame_detections=multi_frame_detections,
            track_to_player=track_to_player,
            players=players
        )

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

def create_track_embeddings(
    tracker: AffineAwareByteTrack,
    train_dir: str,
    device: torch.device,
    model_save_path: Optional[str] = None,
    inference_only: bool = True
) -> None:
    """
    Helper function to create embeddings for tracks using a trained model.
    
    Args:
        tracker: The tracker containing track data
        train_dir: Directory containing training data
        device: Device to run inference on
        model_save_path: Path to saved model (if None, uses train_dir for inference-only)
        inference_only: Whether to load model in inference-only mode
    """
    logger.info("Creating embeddings for tracks using trained model")
    
    # Setup embeddings processor
    emb_proc = EmbeddingsProcessor(
        train_dir=train_dir,
        model_save_path=model_save_path,
        device=device
    )
    emb_proc.setup_model(SiameseNet, inference_only=inference_only)
    
    # Get inference transforms
    inference_transforms = get_transforms('opencv_safe')
    
    # Create wrapper function for embeddings processor
    def embeddings_processor_with_transforms(crops):
        return emb_proc.create_embeddings_from_crops(
            crops=crops,
            transform=inference_transforms
        )
    
    # Generate embeddings for all tracks
    tracker.create_embeddings_for_tracks(
        embeddings_processor=embeddings_processor_with_transforms,
        device=device
    )

def ensure_player_mappings(
    multi_frame_detections: List[sv.Detections],
    track_to_player: Dict[int, Player],
    players: set[Player]
) -> Tuple[set[Player], Dict[int, Player]]:
    """
    Helper function to ensure all tracker IDs in detections have player mappings.
    Creates placeholder players for any missing tracker IDs.
    
    Args:
        multi_frame_detections: List of detections across all frames
        track_to_player: Existing mapping from tracker ID to Player object
        players: Set of existing Player objects
        
    Returns:
        Tuple of (updated players set, updated track_to_player mapping)
    """
    # Collect all tracker IDs from detections
    all_tracker_ids_in_detections = set()
    for detections in multi_frame_detections:
        if detections.tracker_id is not None:
            all_tracker_ids_in_detections.update(detections.tracker_id)
    
    # Find missing tracker IDs
    missing_tracker_ids = all_tracker_ids_in_detections - set(track_to_player.keys())
    
    if missing_tracker_ids:
        logger.warning(f"Found {len(missing_tracker_ids)} tracker IDs in detections without player mappings")
        
        # Create placeholder players for missing tracker IDs
        for tid in missing_tracker_ids:
            placeholder_player = Player(
                tracker_id=tid,
                initial_embedding=None,
                initial_crops=[],
                team=-1  # Unknown team
            )
            players.add(placeholder_player)
            track_to_player[tid] = placeholder_player
            logger.debug(f"Created placeholder player for missing tracker ID {tid}")
        
        logger.info(f"Created {len(missing_tracker_ids)} placeholder players for orphaned tracker IDs")
    
    return players, track_to_player

def run_player_training_pipeline(
    data_dir: str,
    all_crops_dir: str,
    temp_dir: str,
    device: torch.device,
    tracker: AffineAwareByteTrack,
    tracks_data: Dict[int, TrackData],
    multi_frame_detections: List[sv.Detections]
) -> Tuple[set[Player], Dict[int, Player], List[sv.Detections]]:
    """
    Helper function to run the complete player training pipeline including:
    - Track identifier training
    - Clustering
    - Player identifier training
    - Track embeddings creation
    - Player association
    
    Args:
        data_dir: Base data directory for training
        crop_processor: Crop extractor processor instance
        temp_dir: Temporary directory path
        device: PyTorch device for training
        tracker: Tracker instance containing track data
        tracks_data: Dictionary of track data
        multi_frame_detections: List of detections across frames
        
    Returns:
        Tuple of (players set, track_to_player mapping, updated multi_frame_detections)
    """
    logger.info("Starting player training pipeline")
    
    # --- Train Track Identifier ---
    logger.info("Training track identifier model")
    embeddings_model_path = os.path.join(data_dir, "embeddings_model.pth")
    train_dir = os.path.join(data_dir, "train")
    track_train_processor = EmbeddingsProcessor(
        train_dir=train_dir, 
        model_save_path=embeddings_model_path, 
        device=device
    )
    track_train_processor.train_and_save(
        model_class=SiameseNet,
        dataset_class=LacrossePlayerDataset,
        transform=get_transforms('opencv_safe_training')
    )

    # --- Cluster Tracks based on their similarity ---
    logger.info("Clustering tracks based on similarity")
    clustering_processor = ClusteringProcessor(
        model_path=embeddings_model_path,
        all_crops_dir=data_dir,
        clustered_data_dir=os.path.join(data_dir, "clustered"),
        temp_dir=temp_dir,
        device=device
    )

    num_clusters, num_images = clustering_processor.process_clustering_with_search(
        model_class=SiameseNet,
        source_data_dir=all_crops_dir
    )
    
    logger.info(f"Clustering complete: {num_clusters} clusters from {num_images} images")

    # --- Prepare clustered images for retraining ---
    logger.info("Preparing clustered data for player training")
    player_crops_dir = clustering_processor.get_clustered_data_directory()
    player_data_dir = os.path.join(player_crops_dir, "data")
    create_train_val_split(player_crops_dir, player_data_dir)

    # --- Train player identifier ---
    logger.info("Training player identifier model")
    player_embeddings_model_path = os.path.join(player_data_dir, "player_embeddings_model.pth")
    player_train_dir = os.path.join(player_data_dir, "train")
    player_processor = EmbeddingsProcessor(
        train_dir=player_train_dir, 
        model_save_path=player_embeddings_model_path, 
        device=device
    )
    player_processor.train_and_save(
        model_class=SiameseNet,
        dataset_class=LacrossePlayerDataset,
        transform=get_transforms('opencv_safe_training')
    )

    # --- Create embeddings for tracks using the trained model ---
    logger.info("Creating embeddings for tracks using trained player model")
    create_track_embeddings(
        tracker=tracker,
        train_dir=player_train_dir,
        device=device,
        model_save_path=player_embeddings_model_path,
        inference_only=False
    )
    
    # --- Associate tracks to players ---
    logger.info("Associating tracks to players")
    players, track_to_player, updated_multi_frame_detections = associate_tracks_to_players_with_stitching(
        tracks_data, multi_frame_detections
    )
    logger.info(f"Created {len(players)} players for {len(tracks_data)} tracks.")
    
    return players, track_to_player, updated_multi_frame_detections
