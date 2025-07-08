import logging
from typing import Optional, List, Dict, Tuple
import datetime
import os
import torch
import cv2
import supervision as sv
from supervision.annotators.utils import ColorLookup
from tqdm import tqdm
from collections import deque
import numpy as np

from tools import reporting
from modules.detection import DetectionModel
from modules.player import Player
from modules.team_identification import TeamIdentifier, PlayerMasker
from tools.store_driver import Store
from modules.custom_tracker import AffineAwareByteTrack, TrackData
from modules.Siglip_reid import SiglipReID
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

    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    generator_params = {
        "source_path": input_video,
        "end": debug_max_frames
    }
    frames_generator = sv.get_video_frames_generator(**generator_params)
    model = DetectionModel(store=store, device=device)    
    tracker = AffineAwareByteTrack() 
    team_identifier = TeamIdentifier()
    masker = PlayerMasker()
    masker.type = "grass_avg"  # Set the type after initialization
    
    emb_provider = SiglipReID()
    
    ellipse_annotator = sv.EllipseAnnotator()
    label_annotator = sv.LabelAnnotator()
    multi_frame_detections = deque()

    frame_target = debug_max_frames if debug_max_frames else video_info.total_frames



    # --- Generate detections and tracks for each frame ---

    logger.info("Generating detections and tracks for each frame")
    previous_frame = None
    affine_matrix = None

    tracker = AffineAwareByteTrack()
    
    # Collect 5 frames equally spread throughout the video for grass estimation
    sample_frames = []
    frame_indices_to_sample = []
    if frame_target > 0:
        if frame_target >= 5:
            # Sample 5 frames: beginning, quarter, middle, three-quarter, end
            frame_indices_to_sample = [
                0,                              # Beginning
                frame_target // 4,              # Quarter
                frame_target // 2,              # Middle  
                3 * frame_target // 4,          # Three-quarter
                frame_target - 1                # End
            ]
        else:
            # Use all available frames if less than 5
            frame_indices_to_sample = list(range(frame_target))

    current_frame_idx = 0
    for frame in tqdm(frames_generator, desc="Frames read", total=frame_target):
        try:
            # Collect sample frames for grass estimation
            if current_frame_idx in frame_indices_to_sample:
                sample_frames.append(frame.copy())
            
            all_detections = model.generate_detections(frame)
            if all_detections.xyxy.size > 0 and np.any(all_detections.xyxy < 0):
                logger.warning("Detections from model contain negative xyxy coordinates. This may indicate an issue with the detection model.")

            # Affine matrix determines any camera move between consecutive frames (very important)
            if previous_frame is not None:
                affine_matrix = AffineAwareByteTrack.calculate_affine_transform(previous_frame, frame)
            if affine_matrix is None:
                affine_matrix = AffineAwareByteTrack.get_identity_affine_matrix()

            detections = tracker.update_with_transform(
                detections=all_detections,
                frame=frame,
                affine_matrix=affine_matrix
            )

            previous_frame = frame.copy()
           
        except Exception as e:
            logger.error(f"Error during detection/tracking for frame: {e}", exc_info=True)
            multi_frame_detections.append(sv.Detections.empty()) # Append empty detections to keep deque length consistent
            current_frame_idx += 1
            continue

        multi_frame_detections.append(detections)
        current_frame_idx += 1


     # --- Creating Embeddings and Finding Teams for each track  ---

    tracker_data = tracker.get_tracks_data()
    tracker_data = {tid: data for tid, data in tracker_data.items() if data.class_id == _PLAYER_CLASS_ID}
    
    # Extract crops from track data for convenience
    tids = []
    crops = []
    for track_id, track_data in tracker_data.items():
        if track_data.largest_crop is not None:
            tids.append(track_id)
            crops.append(track_data.largest_crop)
    
    logger.info(f"Using {len(sample_frames)} sample frames for grass estimation")
    masked_crops = masker.fit_predict(crops, frames=sample_frames)
    embs: np.ndarray = emb_provider.get_emb_from_crops(masked_crops) # 2D array, 0 for batch, 1 for siglip embedding
    avg_colors = emb_provider.get_avg_color_from_crops(masked_crops)
    teams_array = team_identifier.fit_predict(avg_colors)

    for tid, emb, team in zip(tids, embs, teams_array):
        tracker.update_track_with_embedding_and_team(tid, emb, int(team))

    # Create tracker_crops dictionary for association functions
    tracker_crops: Dict[int, np.ndarray] = {}
    for track_id, track_data in tracker_data.items():
        if track_data.largest_crop is not None:
            tracker_crops[track_id] = track_data.largest_crop


     # --- Analysing Tracks to Create Players  ---

    players: set[Player] = set()  # Set to store unique Player objects
    track_to_player: Dict[int, Player] = {}  # Mapping from tracker ID to Player object
    tracks_data = tracker.get_tracks_data()
    
    # Association parameters
    SIMILARITY_THRESHOLD = 0.9  # Minimum cosine similarity for association
    USE_GLOBAL_OPTIMIZATION = True  # Set to False for greedy algorithm
    
    logger.info(f"Associating {len(tracks_data)} tracks to players using {'global optimization' if USE_GLOBAL_OPTIMIZATION else 'greedy'} with cosine similarity")
    
    if USE_GLOBAL_OPTIMIZATION:
        # Use global optimization approach
        players, track_to_player = associate_tracks_to_players_globally(tracks_data, tracker_crops, SIMILARITY_THRESHOLD)
    else:
        # Use greedy chronological approach
        players, track_to_player = associate_tracks_to_players_greedy(tracks_data, tracker_crops, SIMILARITY_THRESHOLD)

    logger.info(f"Player association complete: {len(players)} unique players from {len(tracks_data)} tracks")

    # --- Write output video using stored detections and validated player info ---

    logger.info("Creating output video with annotated frames")

    writer_frames_generator = sv.get_video_frames_generator(**generator_params)
    with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink:
        for frame in tqdm(writer_frames_generator, desc="Writing frames", total=frame_target):
            try:
                detections: sv.Detections = multi_frame_detections.popleft()
            except IndexError:
                logger.error(f"Detections could not be retrieved to write a frame.")
                break
            if detections.tracker_id is None or detections.confidence is None:
                logger.error(f"Missing tracking ID or confidence in detections.")
                continue
            detections.data["player_id"] = np.zeros(len(detections), dtype=np.int16)
            detections.data["team_id"] = np.full(len(detections), -1, dtype=np.int16)

            # Prepare lists for labels and colors
            labels = []
            colors = []

            #Loops through all detections in a given frame, and assigns a player_id to all confirmed ones.
            for i in range(len(detections)):
                tracker_id_np = detections.tracker_id[i]
                if tracker_id_np is None:
                    detections.data["player_id"][i] = -1
                    detections.data["team_id"][i] = -1
                    labels.append("?")
                    colors.append((64, 64, 64))  # Dark gray for untracked
                    continue
                    
                tid = int(tracker_id_np)
                player = track_to_player.get(tid, None)
                
                if player is None:
                    logger.warning(f"Tracker ID {tid} has no associated Player object. Skipping annotation.")
                    detections.data["player_id"][i] = -1
                    detections.data["team_id"][i] = -1
                    label, color = get_player_display_info(None, -1)
                else:
                    detections.data["player_id"][i] = player.id
                    team = player.team if player.team is not None else -1
                    detections.data["team_id"][i] = team
                    label, color = get_player_display_info(player, team)
                
                labels.append(label)
                colors.append(color)

            # Create annotated frame with team-based colors
            annotated_frame = frame.copy()
            
            # Manually draw annotations with custom colors
            for i in range(len(detections)):
                if detections.xyxy[i] is None:
                    continue
                    
                # Get detection box and color
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                color = colors[i]
                label = labels[i]
                
                # Draw ellipse/circle for detection
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int((x2 - x1) / 2)
                height = int((y2 - y1) / 2)
                
                # Draw ellipse
                cv2.ellipse(annotated_frame, (center_x, center_y), (width, height), 0, 0, 360, color, 2)
                
                # Draw label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, 
                            (center_x - label_size[0]//2 - 5, center_y - label_size[1] - 10),
                            (center_x + label_size[0]//2 + 5, center_y - 5),
                            color, -1)
                cv2.putText(annotated_frame, label, 
                          (center_x - label_size[0]//2, center_y - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            sink.write_frame(frame=annotated_frame)

    # --- (Optional) Fourth pass: Generate analysis report ---
    generate_report = True
    if generate_report:
        logger.info("Generating per-track analysis report")
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(reporting.REPORTS_BASE_DIR, run_id)

        # Collect per-track info: original crop, masked crop, team, player_id, track_id
        per_track_rows = []
        for tid, track_data in tracker_data.items():
            # Find masked crop index
            try:
                crop_idx = tids.index(tid)
                masked_crop = masked_crops[crop_idx]
            except (ValueError, IndexError):
                masked_crop = None
            player = track_to_player.get(tid, None)
            per_track_rows.append({
                "track_id": tid,
                "original_crop": track_data.largest_crop,
                "masked_crop": masked_crop,
                "team": getattr(track_data, 'team', -1),
                "player_id": player.id if player is not None else -1
            })

        logger.info(f"Found {len(per_track_rows)} tracks to report on.")
        reporting.generate_track_report_html(run_id, run_output_dir, per_track_rows)
        reporting.update_main_index_html()

def get_player_display_info(player: Optional[Player], team: int) -> Tuple[str, Tuple[int, int, int]]:
    """
    Get display label and color for a player based on their team and role.
    
    Args:
        player: Player object (can be None for untracked detections)
        team: Team ID from detection
        
    Returns:
        Tuple of (label, color_rgb)
    """
    if player is None:
        return "?", (64, 64, 64)  # Dark gray for untracked
    
    if team == 2:  # Referee
        return "REF", (0, 255, 0)  # Green
    elif team == 10:  # Goalie team 0
        return "GK0", (255, 255, 0)  # Yellow
    elif team == 11:  # Goalie team 1
        return "GK1", (255, 165, 0)  # Orange
    elif team == 0:  # Team 0 player
        return f"P{player.id:02d}", (255, 0, 0)  # Red
    elif team == 1:  # Team 1 player
        return f"P{player.id:02d}", (0, 0, 255)  # Blue
    else:  # Unknown team
        return f"P{player.id:02d}?", (128, 128, 128)  # Gray
