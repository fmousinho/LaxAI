
import logging
from typing import Optional, List
import datetime
import os
import torch
import supervision as sv
from tqdm import tqdm
from collections import deque
import numpy as np

from .tools import reporting
from .modules.detection import DetectionModel
from .modules.player import Player
from .tools.store_driver import Store
from .modules.custom_tracker import AffineAwareByteTrack


logger = logging.getLogger(__name__)


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
    
    #smoother = sv.DetectionsSmoother() #Smoother doesn't produce good results.

    ellipse_annotator = sv.EllipseAnnotator() 
    label_annotator = sv.LabelAnnotator()
    multi_frame_detections = deque()
    multi_ineligible_track_ids = deque(List[int])

    frame_target = debug_max_frames if debug_max_frames else video_info.total_frames

    # --- First pass: Process frames, update/create players, and store all detections ---
    # This loop ensures all Player instances in Player._registry are up-to-date with their
    # confirmation counts and validation status.
    logger.info("First pass: Reading frames and detectiong players")
    previous_frame = None
    affine_matrix = None
    for frame in tqdm(frames_generator, desc="Reading frames and detecting players", total=frame_target):
        try:
            # Detector determines where the players are
            detections = model.generate_detections(frame)
            if detections.xyxy.size > 0 and np.any(detections.xyxy < 0):
                logger.warning("Detections from model contain negative xyxy coordinates. This may indicate an issue with the detection model.")

            # Affine matrix determines any camera move between consecutive frames
            if previous_frame is not None:
                affine_matrix = AffineAwareByteTrack.calculate_affine_transform(previous_frame, frame)
            if affine_matrix is None:
                affine_matrix = AffineAwareByteTrack.get_identity_affine_matrix()
            # The tracker applies the affine matrix to the previous tracked detections
            detections = tracker.update_with_transform(detections, affine_matrix, frame=frame)
            previous_frame = frame.copy()
           
        except Exception as e:
            logger.error(f"Error during detection/tracking for frame: {e}", exc_info=True)
            # If detection or tracking fails for a frame, we might not have valid 'detections'.
            # It's safer to skip processing this frame further for player updates.
            multi_frame_detections.append(sv.Detections.empty()) # Append empty detections to keep deque length consistent
            continue

        multi_frame_detections.append(detections)
        ineligible_tracker_ids = tracker.get_tids_for_frame()
        multi_ineligible_track_ids.append(ineligible_tracker_ids)


    # --- Second pass: Creates players based on video analysis

    logger.info("Second pass: Creating and linking Player objects")
    for frame_idx, detections in enumerate(tqdm(multi_frame_detections, desc="Identifying players", total=len(multi_frame_detections))):
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            continue

        # Collect TIDs that are in the current frame and have re-ID data from SiglipReID,
        # but are NOT yet associated with a Player object in Player._registry.
        tids_for_player_match_or_create = []
        embeddings_for_player_match_or_create = []
        crops_for_player_match_or_create = []

        tracker_ids_ineligible_for_match: List[int] = multi_ineligible_track_ids[frame_idx]

        # Find all tids in the current detections that are not yet linked to a Player object, and collect their re-ID data.
        for tid_np in detections.tracker_id:
            tid = int(tid_np)
            player = Player.get_player_by_tid(tid)
            if player is None:
                reid_data = tracker.get_reid_data_for_tid(tid)
                if reid_data is not None and reid_data[0] is not None and reid_data[1]: # Check if embedding and crops exist
                    tids_for_player_match_or_create.append(tid)
                    embeddings_for_player_match_or_create.append(reid_data[0])
                    crops_for_player_match_or_create.append(reid_data[1])
                else:
                    logger.debug(f"Tracker ID {tid} in frame {frame_idx} has no re-ID data from SiglipReID or data is incomplete. Skipping initial player creation/re-ID for it.")
            else:
                emb, _ = tracker.get_reid_data_for_tid(tid)
                player.update_embeddings(emb)

        # If we have TIDs that need to be matched or created, proceed with the matching process.
        # Matching only happens with TIDs that are considered orphans (not linked to player object nor reassigned.).
        if tids_for_player_match_or_create: 

            # Attempt to match these TIDs to existing (previously removed) players
            unmatched_tids, unmatched_embeddings, unmatched_crops = Player.match_and_update_for_batch(
                new_tracker_ids=tids_for_player_match_or_create,
                new_embeddings=embeddings_for_player_match_or_create,
                new_crops=crops_for_player_match_or_create,
                tracker_ids_ineligible_for_match=tracker_ids_ineligible_for_match
            )


            # For any remaining unmatched TIDs, create new Player objects
            if unmatched_tids:
                Player.register_new_players_batch(
                    tracker_ids=unmatched_tids,
                    embeddings=unmatched_embeddings,
                    crops=unmatched_crops
                )
                logger.debug(f"Created {len(unmatched_tids)} new Player objects for unmatched tracker IDs in frame {frame_idx}.")


    # --- Third pass: Write output video using stored detections and validated player info ---
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
            
            annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=detections)

            labels = [
                f"{pid:.0f}" 
                for pid in detections.data["player_id"]
            ]
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
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
