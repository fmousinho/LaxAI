
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
from .modules.affine_motion_compensation import AffineAwareByteTrack


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

    tracker = AffineAwareByteTrack(  
        track_activation_threshold = 0.1,
        lost_track_buffer = 30,
        minimum_matching_threshold=0.7,
        frame_rate = video_info.fps,
        minimum_consecutive_frames=1
        ) 
    #smoother = sv.DetectionsSmoother() #Smoother doesn't produce good results.

    ellipse_annotator = sv.EllipseAnnotator() 
    label_annotator = sv.LabelAnnotator()
    multi_frame_detections = deque()

    frame_target = debug_max_frames if debug_max_frames else video_info.total_frames

    # --- First pass: Process frames, update/create players, and store all detections ---
    # This loop ensures all Player instances in Player._registry are up-to-date with their
    # confirmation counts and validation status.
    logger.info("Starting first pass: detecting, tracking, and updating player statuses...")
    previous_frame = None
    affine_matrix = None
    for frame in tqdm(frames_generator, desc="Reading and processing frames", total=frame_target):
        try:
            # Detector determines where the players are
            detections = model.generate_detections(frame)
            # Affine matrix determines any camera move between consecutive frames
            if previous_frame is not None:
                affine_matrix = AffineAwareByteTrack.calculate_affine_transform(previous_frame, frame)
            if affine_matrix is None:
                affine_matrix = AffineAwareByteTrack.get_identity_affine_matrix()
            # The tracker applies the affine matrix to the previous tracked detections
            detections = tracker.update_with_transform(detections, affine_matrix)
            previous_frame = frame.copy()
            # Use line bellow instead of the Affine sequence if only the standard tracker is needed.
            # detections = tracker.update_with_detections(detections)
        except Exception as e:
            logger.error(f"Error during detection/tracking for frame: {e}", exc_info=True)
            # If detection or tracking fails for a frame, we might not have valid 'detections'.
            # It's safer to skip processing this frame further for player updates.
            multi_frame_detections.append(sv.Detections.empty()) # Append empty detections to keep deque length consistent
            continue

        # Check if tracker_id is not None before iterating.
        # An empty np.ndarray for tracker_id is fine and the loop won't execute.
        if detections.tracker_id is not None:
            for i in range(len(detections)):
                player_tid = detections.tracker_id[i]
                tid = int(player_tid)
                player, is_new = Player.update_or_create(tracker_id=player_tid)
                if is_new:
                    bbox_xyxy = detections.xyxy[i]
                    player_crop = sv.crop_image(frame, xyxy=bbox_xyxy)
              
                    
        multi_frame_detections.append(detections)

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
                if player is None or not player.is_validated:
                    continue
                detections.data["player_id"][i] = player.id
            
            annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=detections)

            labels = [
                f"{pid:.0f} {confidence:.0%}" 
                for pid, confidence in zip(detections.data["player_id"], detections.confidence)
            ]
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
            sink.write_frame(frame=annotated_frame)
