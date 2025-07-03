import logging
from typing import Optional, List, Dict, Tuple
import datetime
import os
import torch
import supervision as sv
from supervision.annotators.utils import ColorLookup
from tqdm import tqdm
from collections import deque
import numpy as np

from tools import reporting
from modules.detection import DetectionModel
from modules.player import Player
from .modules.team_identification import TeamIdentification
from .tools.store_driver import Store
from .modules.custom_tracker import AffineAwareByteTrack


logger = logging.getLogger(__name__)

_TEAM_COLORS =  {
    0: (255, 0, 0),    # Red for team 0
    1: (0, 0, 255),    # Blue for team 1
    10: (0, 255, 0),   # Green Referees
    20: (255, 255, 0), # Yellow for Goalies 
    -1: (128, 128, 128) # Gray for unknown
}



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
    team_identifier = TeamIdentification()
    
    ellipse_annotator = sv.EllipseAnnotator()
    label_annotator = sv.LabelAnnotator()
    multi_frame_detections = deque()
    multi_ineligible_track_ids: deque[list[int]] = deque()
    trackerid_to_reid_data: Dict[int, Tuple[np.ndarray, int]] = {}

    frame_target = debug_max_frames if debug_max_frames else video_info.total_frames



    # --- Generate detections and tracks for every frames ---

    logger.info("Reading frames and detecting players")
    previous_frame = None
    affine_matrix = None
    for frame in tqdm(frames_generator, desc="Frames read", total=frame_target):
        try:
            detections = model.generate_detections(frame)
            if detections.xyxy.size > 0 and np.any(detections.xyxy < 0):
                logger.warning("Detections from model contain negative xyxy coordinates. This may indicate an issue with the detection model.")

            # Affine matrix determines any camera move between consecutive frames (very important)
            if previous_frame is not None:
                affine_matrix = AffineAwareByteTrack.calculate_affine_transform(previous_frame, frame)
            if affine_matrix is None:
                affine_matrix = AffineAwareByteTrack.get_identity_affine_matrix()
            detections = tracker.update_with_transform(detections, affine_matrix, frame=frame)
            previous_frame = frame.copy()
           
        except Exception as e:
            logger.error(f"Error during detection/tracking for frame: {e}", exc_info=True)
            multi_frame_detections.append(sv.Detections.empty()) # Append empty detections to keep deque length consistent
            continue

        multi_frame_detections.append(detections)
        ineligible_tracker_ids = tracker.get_tids_for_frame()
        multi_ineligible_track_ids.append(ineligible_tracker_ids)

        # Gets embeddings for all tracker Ids. It may get overwritten in the next pass, but that's okay.
        if detections.tracker_id is not None:
            for tid_np in detections.tracker_id:
                tid = int(tid_np)
                reid_data = tracker.get_reid_data_for_tid(tid)
                if reid_data is not None:
                    embedding, _, det_class = reid_data
                    trackerid_to_reid_data[tid] = (embedding, det_class)

    # --- Team Identification Step ---

    logger.info("Discovering teams based on tracker embeddings.")

    if not trackerid_to_reid_data:
        raise ValueError("No tracker IDs with re-ID data found. Ensure the detection model provides valid embeddings.")
    tracker_id_to_team_mapping = team_identifier.discover_teams(trackerid_to_reid_data)


     # --- Creating Players Step ---

    logger.info("Creating and linking Player objects")

    for frame_idx, detections in enumerate(tqdm(multi_frame_detections, desc="Identifying players", total=len(multi_frame_detections))):
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            continue

        tids_for_player_match_or_create = [] # Tids that are not yet linked to a Player
        embeddings_for_player_match_or_create = []
        crops_for_player_match_or_create = []
        det_classes_for_player_match_or_create = []

        # Tracker Ids in a frame can't be used to match a new Tid
        tracker_ids_ineligible_for_match: List[int] = multi_ineligible_track_ids[frame_idx]

        for tid_np in detections.tracker_id:
            tid = int(tid_np)
            player = Player.get_player_by_tid(tid)
            if player is None:
                reid_data = tracker.get_reid_data_for_tid(tid) 
                if reid_data is not None:
                    embedding, crops, det_class = reid_data
                    tids_for_player_match_or_create.append(tid)
                    embeddings_for_player_match_or_create.append(embedding)
                    crops_for_player_match_or_create.append(crops)
                    det_classes_for_player_match_or_create.append(det_class)
                else:
                    logger.debug(f"Tracker ID {tid} in frame {frame_idx} has no re-ID data from SiglipReID or data is incomplete. Skipping initial player creation/re-ID for it.")
            else:
                # If a player exists for the tracker ID, we simply update their embeddings.
                reid_data = tracker.get_reid_data_for_tid(tid)
                if reid_data:
                    emb, _, det_class = reid_data
                    player.update_embeddings(emb, new_det_class=det_class)

       # Now that we have collected all TIDs that need to be matched or created, we proceed with the matching and creation process.
        if tids_for_player_match_or_create: 

            unmatched_tids, unmatched_embeddings, unmatched_crops, unmatched_teams = Player.match_and_update_for_batch(
                new_tracker_ids=tids_for_player_match_or_create,
                new_embeddings=embeddings_for_player_match_or_create,
                new_crops=crops_for_player_match_or_create,
                new_tid_teams=[tracker_id_to_team_mapping.get(tid, -1) for tid in tids_for_player_match_or_create],
                tracker_ids_ineligible_for_match=tracker_ids_ineligible_for_match
            )

            # For any remaining unmatched TIDs, create new Player objects
            if unmatched_tids:
                Player.register_new_players_batch(
                    tracker_ids=unmatched_tids,
                    embeddings=unmatched_embeddings,
                    crops=unmatched_crops,
                    teams=unmatched_teams
                )
                logger.debug(f"Created {len(unmatched_tids)} new Player objects for unmatched tracker IDs in frame {frame_idx}.")
  

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

            #Loops through all detections in a given frame, and assigns a player_id to all confirmed ones.
            for i in range(len(detections)):
                tracker_id_np = detections.tracker_id[i]
                if tracker_id_np is None:
                    continue
                tid = int(tracker_id_np)
                player = Player.get_player_by_tid(tid)
                if player is None:
                    logger.warning(f"Tracker ID {tid} in frame {frame_idx} has no associated Player object. Skipping annotation.")
                    detections.data["player_id"][i] = -1  # Assign -1 for untracked players
                else:
                    detections.data["player_id"][i] = player.id
                    team = player.team if player.team is not None else -1

            labels = []
            for i, pid in enumerate(detections.data["player_id"]):
                tid = int(detections.tracker_id[i])
                player = Player.get_player_by_tid(tid)
                team = player.team if player and player.team is not None else -1
                if team in [0, 1]:
                    label = f"P:{pid:.0f} T:{team}"
                elif team == 2:
                    label = "ref"
                else:
                    label = "?"
                labels.append(label)

            annotated_frame = ellipse_annotator.annotate(
                scene=frame.copy(),
                detections=detections,
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
            )
            sink.write_frame(frame=annotated_frame)

     # --- (Optional) Fourth pass: Generate analysis report ---
    if generate_report:
        logger.info("Generating player analysis report")
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(reporting.REPORTS_BASE_DIR, run_id)
        
        unique_players = list(set(Player._registry.values()))
        logger.info(f"Found {len(unique_players)} unique players to report on.")

        reporting.generate_player_report_html(run_id, run_output_dir, unique_players)
        reporting.update_main_index_html()

    # --- (Optional) Fourth pass: Generate analysis report ---
    if generate_report:
        logger.info("Generating player analysis report")
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(reporting.REPORTS_BASE_DIR, run_id)
        
        unique_players = list(set(Player._registry.values()))
        logger.info(f"Found {len(unique_players)} unique players to report on.")

        reporting.generate_player_report_html(run_id, run_output_dir, unique_players)