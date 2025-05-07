import os
import logging
import constants as const
import torch
from store_driver import Store
import ipywidgets as widgets
from videotools import VideoToools
import cv2
from model import VideoModel
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_FILE = "FCA_Upstate_NY_003.mp4"

logger = logging.getLogger(__name__)

def check_for_gpu() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the first available GPU
        logger.info("GPU is available")
    else:
        device = torch.device("cpu")  # Use the CPU if no GPU is available
        logger.info("GPU is not available, using CPU instead.")
    return device

def run_application(store: Store):

    # --- Check for GPU availability ---
    device = check_for_gpu() 

    # --- Model Loading ---
    video_model = VideoModel(model_name=const.MODEL_NAME, drive_path=const.GOOGLE_DRIVE_PATH, device=device)
    if not video_model.load_from_drive(store):
        return

    # --- Download video to be processed ---
    logger.info(f"Downloading video file: {VIDEO_FILE}")
    temp_video_path = store.download_file_to_temp(VIDEO_FILE, const.GOOGLE_DRIVE_PATH, suffix=".mp4")
    if not temp_video_path:
        return

    tools = None 
    try:
        tools = VideoToools(input_video_path=temp_video_path, output_video_path="results.mp4")
        max_frames = 100
        frame_idx = 0
        for frame in tools.get_next_frame():
            #frame outputs RGB, which is required by model
            detections = video_model.generate_detections(frame)

            defined_teams = {} # Initialize empty
            if len(detections) > 20:
                # define_teams expects RGB frame, which 'frame' currently is
                defined_teams = video_model.define_teams(frame, detections)
            



            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            tracks = video_model.generate_tracks(frame_bgr, detections)
            
            frame_idx += 1
         




            output_frame = tools.draw_tracks(frame, tracks)
            tools.out.write(output_frame)
            logger.info(f"Processed frame {frame_idx}/{tools.in_n_frames} ({frame.shape[1]}x{frame.shape[0]})")
            
            if frame_idx >= max_frames:
                break

        logger.info(f"Video processing completed.")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error initializing VideoTools: {e}")
    finally:
        # Ensure resources are released even if an error occurs during processing
        if tools and tools.cap: tools.cap.release() # Release via tools.cap
        if tools and tools.out: tools.out.release() # Release via tools.out
        logger.info("Video capture and writer resources released.")

    return
