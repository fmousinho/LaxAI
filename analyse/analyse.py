import logging
import os
import datetime
from typing import List, Optional
import cv2
import torch
import supervision as sv
from tqdm import tqdm
from collections import deque
import numpy as np

# Local Application/Library
from core.common.detection import DetectionModel
from modules.player import Player
from tools.store_driver import Store
from modules.tracker import AffineAwareByteTrack

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_BASE_DIR = os.path.join(SCRIPT_DIR, "reports")


def _generate_player_report_html(run_id: str, run_output_dir: str, players: List[Player]):
    """Generates an HTML report summarizing each unique player."""
    os.makedirs(run_output_dir, exist_ok=True)
    crops_dir = os.path.join(run_output_dir, "crops")
    report_html_path = os.path.join(run_output_dir, "report.html")

    # Save all player crops to disk and store their relative paths for the report
    for player in players:
        player_crop_dir = os.path.join(crops_dir, f"player_{player.id}")
        os.makedirs(player_crop_dir, exist_ok=True)
        
        # Temporarily attach the list of saved crop paths to the player object
        player.report_crop_paths = []
        for i, crop_np in enumerate(player.crops):
            if crop_np.size == 0:
                continue
            crop_filename = f"crop_{i}.png"
            crop_abs_path = os.path.join(player_crop_dir, crop_filename)
            # Crops are stored as BGR numpy arrays, so direct imwrite is fine
            cv2.imwrite(crop_abs_path, crop_np)
            
            # Path relative to the report.html file
            relative_path = os.path.join("crops", f"player_{player.id}", crop_filename)
            player.report_crop_paths.append(relative_path)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Player Analysis Report - Run {run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 3px #ccc; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #e9e9e9; }}
        td:nth-child(1) {{ width: 10%; font-weight: bold; text-align: center; }}
        td:nth-child(2) {{ width: 20%; }}
        .crop-gallery {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .crop-gallery img {{ max-width: 80px; max-height: 80px; border: 1px solid #eee; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Player Analysis Report - Run ID: {run_id}</h1>
    
    <table>
        <thead>
            <tr>
                <th>Player ID</th>
                <th>Associated Tracker IDs</th>
                <th>Player Crops</th>
            </tr>
        </thead>
        <tbody>
"""
    # Sort players by ID for a consistent report order
    sorted_players = sorted(players, key=lambda p: p.id)

    for player in sorted_players:
        tracker_ids_str = ", ".join(map(str, sorted(player.associated_tracker_ids)))
        
        html_content += f"""
            <tr>
                <td>{player.id}</td>
                <td>{tracker_ids_str}</td>
                <td>
                    <div class="crop-gallery">
"""
        for crop_path in player.report_crop_paths:
            html_content += f'                        <img src="{crop_path}" alt="Crop for player {player.id}">\n'
        
        html_content += """
                    </div>
                </td>
            </tr>
"""

    html_content += """
        </tbody>
    </table>
</body>
</html>
"""
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Generated Player Analysis HTML report for run {run_id} at {report_html_path}")


def _update_main_index_html(reports_base_dir: str):
    """Generates or updates the main index.html listing all runs."""
    os.makedirs(reports_base_dir, exist_ok=True)
    index_html_path = os.path.join(reports_base_dir, "index.html")

    run_ids = []
    if os.path.exists(reports_base_dir):
        for item in os.listdir(reports_base_dir):
            item_path = os.path.join(reports_base_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "report.html")):
                run_ids.append(item)
    
    run_ids.sort(reverse=True)

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Runs</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        ul { list-style-type: none; padding: 0; }
        li { background-color: #fff; margin: 10px 0; padding: 15px; border-radius: 5px; box-shadow: 0 2px 3px #ccc; }
        a { text-decoration: none; color: #007bff; font-weight: bold; font-size: 1.1em; }
        a:hover { text-decoration: underline; color: #0056b3; }
    </style>
</head>
<body>
    <h1>Available Analysis Runs</h1>
"""
    if not run_ids:
        html_content += "    <p>No analysis runs found.</p>\n"
    else:
        html_content += "    <ul>\n"
        for run_id in run_ids:
            html_content += f'        <li><a href="{run_id}/report.html">Run - {run_id}</a></li>\n'
        html_content += "    </ul>\n"
    
    html_content += """
</body>
</html>
"""
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Updated main index HTML at {index_html_path}")


def analyse_video(
        store: Store,
        input_video: str,
        start_frame: int,
        end_frame: int,
        device: torch.device = torch.device("cpu"),
    ) -> str:
    """
    Analyses a video by running the full detection and re-identification pipeline,
    then generates an HTML report summarizing each unique player found.

    Args:
        store: The Store instance for file management (used for model loading).
        input_video: Path to the input video file.
        start_frame: The starting frame number for processing.
        end_frame: The ending frame number for processing.
        device: The torch device to use for processing.

    Returns:
        str: Path to the generated HTML report for the run.
    """
    # --- 1. Setup and State Reset ---
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(REPORTS_BASE_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    logger.info(f"Starting analysis run {run_id}. Output will be in {run_output_dir}")

    # Reset Player class state to ensure a clean run
    Player._registry.clear()
    Player._next_id = 1

    # --- 2. Run Processing Pipeline (mimics application.py) ---
    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    generator_params = {"source_path": input_video, "start": start_frame, "end": end_frame}
    frames_generator = sv.get_video_frames_generator(**generator_params)
    
    model = DetectionModel(device=device)
    tracker = AffineAwareByteTrack(
        track_activation_threshold=0.1,
        lost_track_buffer=30,
        minimum_matching_threshold=0.7,
        frame_rate=video_info.fps,
        minimum_consecutive_frames=30
    )
    
    multi_frame_detections = deque()
    frame_target = end_frame - start_frame

    # --- Pass 1: Detection and Tracking ---
    logger.info("Analysis Pass 1: Detecting and tracking objects...")
    previous_frame = None
    for frame in tqdm(frames_generator, desc="Detecting and Tracking", total=frame_target):
        detections = model.generate_detections(frame)
        affine_matrix = AffineAwareByteTrack.calculate_affine_transform(previous_frame, frame) if previous_frame is not None else AffineAwareByteTrack.get_identity_affine_matrix()
        detections = tracker.update_with_transform(detections, affine_matrix, frame=frame)
        previous_frame = frame.copy()
        multi_frame_detections.append(detections)

    # --- Pass 2: Player Linking and Re-identification ---
    logger.info("Analysis Pass 2: Linking players and performing re-identification...")
    for frame_idx, detections in enumerate(tqdm(multi_frame_detections, desc="Linking Players", total=len(multi_frame_detections))):
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            continue

        tids_to_process = [int(tid) for tid in detections.tracker_id if Player.get_player_by_tid(int(tid)) is None]
        if not tids_to_process:
            continue

        embeddings_to_process = []
        crops_to_process = []
        valid_tids_for_processing = []

        for tid in tids_to_process:
            reid_data = tracker.get_reid_data_for_tid(tid)
            if reid_data and reid_data[0] is not None and reid_data[1]:
                valid_tids_for_processing.append(tid)
                embeddings_to_process.append(reid_data[0])
                crops_to_process.append(reid_data[1])

        if not valid_tids_for_processing:
            continue

        removed_track_ids = tracker.get_removed_track_ids()
        unmatched_tids, unmatched_embeddings, unmatched_crops, _ = Player.match_and_update_for_batch(
            new_tracker_ids=valid_tids_for_processing,
            new_embeddings=embeddings_to_process,
            new_crops=crops_to_process,
            orphan_track_ids=removed_track_ids
        )

        if unmatched_tids:
            Player.register_new_players_batch(
                tracker_ids=unmatched_tids,
                embeddings=unmatched_embeddings,
                crops=unmatched_crops
            )

    # --- 3. Generate Report ---
    unique_players = list(set(Player._registry.values()))
    logger.info(f"Found {len(unique_players)} unique players to report on.")

    _generate_player_report_html(run_id, run_output_dir, unique_players)
    _update_main_index_html(REPORTS_BASE_DIR)

    report_path = os.path.join(run_output_dir, 'report.html')
    logger.info(f"Analysis run {run_id} completed. Report at: {report_path}")
    return report_path
      
