
import logging
from typing import Optional, List
import torch
import supervision as sv
from tqdm import tqdm
from collections import deque
import numpy as np

from .modules.detection import DetectionModel
from .modules.player import Player
from .tools.store_driver import Store

logger = logging.getLogger(__name__)


def run_application (
        store: Store,
        input_video: str,
        output_video_path: str = "results.mp4",
        device: torch.device = torch.device("cpu"),
        debug_max_frames: Optional[int] = None
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

    tracker = sv.ByteTrack(  
        track_activation_threshold = 0.25,
        lost_track_buffer = 30,
        minimum_matching_threshold=0.8,
        frame_rate = video_info.fps,
        minimum_consecutive_frames=1
        ) 
    #smoother = sv.DetectionsSmoother()

    ellipse_annotator = sv.EllipseAnnotator() 
    label_annotator = sv.LabelAnnotator()
    multi_frame_detections = deque()

    frame_id = 0
    frame_target = debug_max_frames if debug_max_frames else video_info.total_frames

    # --- First pass: Process frames, update/create players, and store all detections ---
    # This loop ensures all Player instances in Player._registry are up-to-date with their
    # confirmation counts and validation status.
    logger.info("Starting first pass: detecting, tracking, and updating player statuses...")
    for frame in tqdm(frames_generator, desc="Reading and processing frames", total=frame_target):
        detections = model.generate_detections(frame)
        detections = tracker.update_with_detections(detections)

        # Assign player_id based on tracker_id
        if detections.tracker_id is not None:
            player_ids_list = [
                Player.update_or_create(int(tid)) if tid is not None else 0
                for tid in detections.tracker_id
            ]
        multi_frame_detections.append(detections)
        frame_id +=1

    # --- Second pass: Write output video using stored detections and validated player info ---
    logger.info("Starting second pass: Annotating and writing video...")

    writer_frames_generator = sv.get_video_frames_generator(**generator_params)
    with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink:
        for frame in tqdm(writer_frames_generator, desc="Writing frames", total=frame_target):
            try:
                detections: sv.Detections = multi_frame_detections.popleft()
            except IndexError:
                logger.error(f"Detections could not be retrieved to write a frame.")
                break
            validated_detections = []
            if detections.tracker_id is None:
                continue
            detections.data["player_id"] = np.zeros(len(detections), dtype=np.int16)

            #Loops through all detections in a given frame, and assigns a player_id to all confirmed ones.
            for i in range(len(detections)):
                tracker_id_np = detections.tracker_id[i]
                if tracker_id_np is None:
                    continue
                tid = int(tracker_id_np)
                player = Player.get_player(tid)
                if player is None or not player.is_validated:
                    continue
                detections.data["player_id"][i] = player.id  
            
            annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=detections)
            if detections.confidence is not None:
                labels = [
                    f"{tracker_id} {confidence:.2f}" 
                    for tracker_id, confidence in zip(detections.data["player_id"], detections.confidence)
                ]
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )
            sink.write_frame(frame=annotated_frame)
