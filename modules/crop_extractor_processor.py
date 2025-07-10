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

        all_detections = self.all_detections.copy()

        logger.info(f"Starting crop extraction.")
        logger.info(f"directory: {all_crops_dir}")
        
        frame_idx = 0
        all_detections_dq = deque(all_detections)

        # Calculate total frames for progress logging
        total_frames = len(all_detections) if all_detections else 0
        total_crops_extracted = 0

        for frame in self.frame_generator:
            log_progress(logger, "Processing frames for crop extraction", frame_idx, total_frames, step=10)

            detections = all_detections_dq.popleft()

            # Skip if no detections
            if len(detections) == 0:
                frame_idx += 1
                continue
            
            # Extract crops for each detection in this frame
            for i in range(len(detections)):
                
                if detections.data is not None and "frame_id" in detections.data and len(detections.data["frame_id"]) > i:
                    frame_id = detections.data["frame_id"][i]
                else:
                    frame_id = frame_idx  # Use current frame index as fallback
                    
                bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
                if detections.tracker_id is not None and len(detections.tracker_id) > i:
                    tracker_id = detections.tracker_id[i]
                else:
                    tracker_id = None

                if detections.confidence is not None and len(detections.confidence) > i:
                    confidence = detections.confidence[i]
                else:
                    confidence = 0.0

                if tracker_id is None:
                    continue
                
                # Extract crop from frame
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                
               # Ensure the directory for this tracker_id exists
                tracker_dir = os.path.join(all_crops_dir, str(tracker_id))
                os.makedirs(tracker_dir, exist_ok=True)

                # Save crop with filename: frame_id_tracker_id_confidence.jpg
                crop_filename = f"{frame_id}_{tracker_id}_{confidence:.3f}.jpg"
                crop_path = os.path.join(tracker_dir, crop_filename)
                cv2.imwrite(crop_path, crop)
                total_crops_extracted += 1
            
            # Update for next frame
            if len(all_detections_dq) > 0:
                next_detection = all_detections_dq[0]
                if next_detection.data is not None and "frame_id" in next_detection.data:
                    next_detected_frame = next_detection.data["frame_id"][0]
                else:
                    next_detected_frame = frame_idx + 1  
            else:
                break
            frame_idx += 1

            log_progress(logger, "Processing frames for crop extraction", frame_idx, total_frames, step=1)

        if total_crops_extracted == 0:
            logger.critical("")
            raise RuntimeError("No crops were extracted. Check your detections and video frames.")

        logger.info(f"Crop extraction complete! {total_crops_extracted} crops extracted.")

    def get_crops_directory (self) -> str:
        return self.crops_dir
    
    def get_all_crops_directory(self) -> str:
        """
        Returns the directory where all crops are stored.
        """
        return self.all_crops_dir


def create_train_val_split(source_folder: str, destin_folder: str, train_ratio: float = 0.8):
    """
    Splits the extracted crops into training and validation sets, maintaining per-track structure.
    
    Args:
        source_folder: Directory containing the source crops folders (each folder contains images)
        destin_folder: Directory where train and val folders will be created
        train_ratio: Ratio of data to use for training (default: 0.8)
    """
    logger.info(f"Creating train/val split from {source_folder} to {destin_folder}")

    # Verify source directory exists
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source crops directory does not exist: {source_folder}")

    # Create train and val directories in destination folder
    train_dir = os.path.join(destin_folder, "train")
    val_dir = os.path.join(destin_folder, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    random.seed(42)  # For reproducibility

    # For each tracker_id directory in source_folder
    for track_id in os.listdir(source_folder):
        track_path = os.path.join(source_folder, track_id)
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
    logger.info(f"Train directory: {train_dir}")
    logger.info(f"Validation directory: {val_dir}")


