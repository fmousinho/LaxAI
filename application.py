
import logging
from . import constants as const
import numpy as np
import torch
from .store_driver import Store
import cv2
import random
from .videotools import VideoToools, BoundingBox
from .model import VideoModel
from typing import Callable, Optional
from torch.utils.tensorboard import SummaryWriter

#When debug mode is on, the following will be used. 
DEBUG_N_FRAMES = 300 #  Procees the first N frames instead of whole video
VIDEO_FILE = "GRIT Dallas-Houston 2027 vs Urban Elite 2027 - 12-30pm.mp4"

logger = logging.getLogger(__name__)

def _sample_frame_indices(num_total_frames: int, num_frames_to_sample: int) -> list[int]:
    """
    Selects a list of frame indices to sample from a video.
    The sampling is biased towards the middle of the video.

    Args:
        num_total_frames: The total number of frames in the video.
        num_frames_to_sample: The target number of frames to sample.

    Returns:
        A sorted list of unique frame indices to be sampled.
    """
    target_indices_to_sample = set()

    if num_total_frames <= num_frames_to_sample:
        target_indices_to_sample.update(range(num_total_frames))
        logger.info(f"Targeting all {num_total_frames} frames for sampling as it's less than or equal to target {num_frames_to_sample}.")
    else:
        middle_start_idx = num_total_frames // 5
        middle_end_idx = num_total_frames - (num_total_frames // 5)
        middle_indices_pool = list(range(middle_start_idx, middle_end_idx))
        if not middle_indices_pool:
            middle_indices_pool = list(range(num_total_frames))

        num_from_middle = min(len(middle_indices_pool), int(num_frames_to_sample * 0.75))
        if middle_indices_pool and num_from_middle > 0:
            target_indices_to_sample.update(random.sample(middle_indices_pool, num_from_middle))

        num_remaining_to_sample = num_frames_to_sample - len(target_indices_to_sample)
        if num_remaining_to_sample > 0:
            available_for_overall_sample = [i for i in range(num_total_frames) if i not in target_indices_to_sample]
            num_overall_additional = min(len(available_for_overall_sample), num_remaining_to_sample)
            if available_for_overall_sample and num_overall_additional > 0:
                target_indices_to_sample.update(random.sample(available_for_overall_sample, num_overall_additional))
    
    return sorted(list(target_indices_to_sample))

def _initialize_team_identifier(
    video_model: VideoModel,
    tools: VideoToools, 
    num_frames_to_sample: int = 20,
    min_detections_for_sampling_frame: int = 10, # Min detections for a frame to be considered for sampling
    writer: Optional[SummaryWriter] = None
) -> Callable[[BoundingBox, np.ndarray], Optional[int]]:
    """
    Samples frames from the video, generates detections, and uses them
    to define teams, returning a team_id_getter.

    Args:
        video_model: The VideoModel instance.
        tools: The VideoToools instance for accessing video properties like total frames.
               and fetching specific frames by index.
        num_frames_to_sample: The target number of frames to sample for team identification.
        min_detections_for_sampling_frame: Minimum detections a frame must have to be included in the sample.

    Returns:
        A callable function that maps a BoundingBox and the current frame
        to a team ID, or a default getter if team identification fails.
    """
    logger.info(f"Attempting to define teams by sampling up to {num_frames_to_sample} frames...")
   
    num_total_frames = tools.in_n_frames
    if num_total_frames == 0:
        logger.warning("Video appears to have no frames (total_frames is 0). Proceeding with default team getter.")
        return video_model.get_default_team_getter()

    # Get the sorted list of frame indices to sample
    sorted_target_indices = _sample_frame_indices(num_total_frames, num_frames_to_sample)
    logger.info(f"Identified {len(sorted_target_indices)} target frame indices for sampling: {sorted_target_indices}")

    sampled_frames_data = []

    for frame_idx_to_process in sorted_target_indices:
        if len(sampled_frames_data) >= num_frames_to_sample:
            logger.info(f"Collected enough ({len(sampled_frames_data)}) valid samples for team identification before processing all targets.")
            break
        
        frame = tools.get_frame_by_index(frame_idx_to_process)
        if frame is None:
            logger.warning(f"Could not retrieve frame at index {frame_idx_to_process} for team ID sampling. Skipping.")
            continue
       
        detections = video_model.generate_detections(frame) 
        if len(detections) >= min_detections_for_sampling_frame:
            sampled_frames_data.append((frame, detections))
            logger.debug(f"Collected frame {frame_idx_to_process} (had {len(detections)} detections) for team ID. {len(sampled_frames_data)} collected.")
        else:
            logger.debug(f"Skipped frame {frame_idx_to_process} for team ID sample (had {len(detections)} detections, need >= {min_detections_for_sampling_frame}).")

    if not sampled_frames_data:
        logger.warning("No frames were sampled with enough detections. Proceeding with default team getter.")
        return video_model.get_default_team_getter()

    logger.info(f"Attempting to identify teams using {len(sampled_frames_data)} sampled frames.")

    return video_model.identifies_team(sampled_frames_data)

def run_application(store: Store, writer: Optional[SummaryWriter] = None, 
                    device: torch.device = torch.device("cpu")) -> None:

    # --- Model Loading ---
    video_model = VideoModel(model_name=const.MODEL_NAME, drive_path=const.GOOGLE_DRIVE_PATH, device=device, writer=writer)
    if not video_model.load_from_drive(store):
        return

    # --- Download video to be processed ---
    logger.info(f"Downloading video file: {VIDEO_FILE}")
    temp_video_path = store.download_file_to_temp(VIDEO_FILE, const.GOOGLE_DRIVE_PATH, suffix=".mp4")
    if not temp_video_path:
        return

    tools = None
    team_id_getter = video_model.get_default_team_getter() 

    #--- Main Logic ---

    try:
        tools = VideoToools(input_video_path=temp_video_path, output_video_path="results.mp4")
        team_id_getter = _initialize_team_identifier(video_model, tools, writer=writer)

        # --- Pipeline to detect and track players, draw frame ---
     
        main_processing_generator = tools.get_next_frame()
        
        if logger.isEnabledFor(logging.DEBUG):
            max_frames = DEBUG_N_FRAMES
            logger.debug(f"DEBUG mode: Processing a maximum of {max_frames} frames.")
        else:
            max_frames = tools.in_n_frames
    
        frame_idx = 0

        logger.info("Starting main video processing loop...")
        for frame_rgb in main_processing_generator: # frame is RGB
            if frame_idx >= max_frames:
                logger.info(f"Reached max_frames limit of {max_frames}.")
                break

            # All model operations use the RGB frame
            detections = video_model.generate_detections(frame_rgb)
            tracks = video_model.generate_tracks(frame_rgb, detections) # Pass RGB frame
            
            # Update the VideoModel's internal track_to_team_map
            # The `team_id_getter` here is the one from _initialize_team_identifier (raw classifications)
            video_model.update_track_team_assignments(tracks, frame_rgb, team_id_getter, current_frame_index=frame_idx)

            # draw_tracks takes RGB frame, and the team_id_getter. It returns an RGB frame.
            # Pass the new method from VideoModel that uses the smoothed map
            output_frame_rgb = tools.draw_tracks(frame_rgb, tracks, team_id_getter=video_model.get_smoothed_team_id_from_map)

            # Convert the final drawn frame to BGR for OpenCV VideoWriter
            output_frame_bgr = cv2.cvtColor(output_frame_rgb, cv2.COLOR_RGB2BGR)
            tools.out.write(output_frame_bgr)
            
            logger.debug(f"Processed frame {frame_idx + 1}/{tools.in_n_frames or 'unknown'} ({frame_rgb.shape[1]}x{frame_rgb.shape[0]})")
            frame_idx += 1

        logger.info(f"Video processing completed.")

    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error initializing VideoTools: {e}")
    finally:
        # Ensure resources are released even if an error occurs during processing
        if tools and tools.cap: tools.cap.release() # Release via tools.cap
        if tools and tools.out: tools.out.release() # Release via tools.out
        logger.info("Video capture and writer resources released.")

    return
