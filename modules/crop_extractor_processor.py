import os
import cv2
from collections import deque
import shutil
import random
import supervision as sv
import logging
from .utils import log_progress

logger = logging.getLogger(__name__)

class CropExtractor:
    """
    Extracts crops from video frames based on detections and organizes them by track_id.
    Each crop is saved in a directory named after the track_id.
    """
    
    def __init__(self, frame_generator, all_detections: list[sv.Detections], temp_dir: str = "temp"):
        self.frame_generator = frame_generator
        self.all_detections = all_detections
        self.temp_dir = temp_dir


    def extract_crops(self):
        """
        Extracts crops from the video frames based on detections and saves them in organized directories.
        """
        # Ensure crops and all_crops directories exist under temp_dir
        crops_dir = os.path.join(self.temp_dir, "crops")
        all_crops_dir = os.path.join(crops_dir, "all_crops")
        os.makedirs(all_crops_dir, exist_ok=True)

        self.crops_dir = crops_dir
        self.all_crops_dir = all_crops_dir

        all_detections = self.all_detections
        frame_generator = self.frame_generator

        logger.info(f"Starting crop extraction.")
        logger.info(f"directory: {all_crops_dir}")
        
        frame_idx = 0
        next_detected_frame = all_detections[0].data["frame_id"][0] if all_detections else 0
        all_detections_dq = deque(all_detections)

        # Calculate total frames for progress logging
        total_frames = len(all_detections) if all_detections else 0
        processed_frames = 0
        total_crops_extracted = 0

        for frame in frame_generator:
            log_progress(logger, "Processing frames for crop extraction", processed_frames, total_frames, step=10)
            if frame_idx != next_detected_frame:
                frame_idx += 1
                continue
            
            detections = all_detections_dq.popleft()
            
            # Extract crops for each detection in this frame
            for i in range(len(detections.xyxy)):
                frame_id = detections.data["frame_id"][i]
                bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
                tracker_id = detections.tracker_id[i]
                confidence = detections.confidence[i]
                
                if tracker_id is None:
                    continue
                
                # Extract crop from frame
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                
                # Save crop with filename: frame_id_confidence.jpg
                crop_filename = f"{frame_id}_{tracker_id}_{confidence:.3f}.jpg"
                crop_path = os.path.join(all_crops_dir, crop_filename)
                cv2.imwrite(crop_path, crop)
                total_crops_extracted += 1
            
            # Update for next frame
            if len(all_detections_dq) > 0:
                next_detected_frame = all_detections_dq[0].data["frame_id"][0]
            else:
                break
            frame_idx += 1
            processed_frames += 1

            log_progress(logger, "Processing frames for crop extraction", processed_frames, total_frames, step=1)

        logger.info(f"Crop extraction complete! {total_crops_extracted} crops extracted.")

    def get_data_directory (self) -> str:
        return self.crops_dir


    def create_train_val_split(self, train_ratio: float = 0.8):
        """
        Splits the extracted crops into training and validation sets, maintaining per-track structure.
        """
        logger.info(f"Creating train/val split for crops.")

        # Create train and val directories
        train_dir = os.path.join(self.crops_dir, "train")
        val_dir = os.path.join(self.crops_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        random.seed(42)  # For reproducibility

        # For each tracker_id directory in crops/
        for track_id in os.listdir(self.all_crops_dir):
            track_path = os.path.join(self.all_crops_dir, track_id)
            if not os.path.isdir(track_path):
                continue

            # List all crop files for this track
            crop_files = [f for f in os.listdir(track_path) if f.endswith('.jpg')]
            random.shuffle(crop_files)

            split_idx = int(train_ratio * len(crop_files))
            train_files = crop_files[:split_idx]
            val_files = crop_files[split_idx:]

            # Create per-track folders in train/ and val/
            train_track_dir = os.path.join(train_dir, track_id)
            val_track_dir = os.path.join(val_dir, track_id)
            os.makedirs(train_track_dir, exist_ok=True)
            os.makedirs(val_track_dir, exist_ok=True)

            # Copy files
            for fname in train_files:
                src = os.path.join(track_path, fname)
                dst = os.path.join(train_track_dir, fname)
                shutil.copy2(src, dst)

            for fname in val_files:
                src = os.path.join(track_path, fname)
                dst = os.path.join(val_track_dir, fname)
                shutil.copy2(src, dst)

        logger.info(f"Crops split with {train_ratio*100:.0f}% for training and {100 - train_ratio*100:.0f}% for validation.")


