import os
import cv2
from collections import deque
import shutil
import random
import supervision as sv
import logging
import numpy as np
from .utils import log_progress

logger = logging.getLogger(__name__)

from config.transforms_config import detection_config

class CropExtractor:
    """
    Extracts crops from video frames based on detections and organizes them by track_id.
    Each crop is saved in a directory named after the track_id.
    """
    
    def __init__(self, frame_generator, all_detections: list[sv.Detections], temp_dir: str, crop_extract_interval: int = detection_config.crop_extract_interval):
        self.frame_generator = frame_generator
        self.all_detections = all_detections
        self.temp_dir = temp_dir
        self.crop_extract_interval = crop_extract_interval


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

        crop_sizes = []
        crop_widths = []
        crop_heights = []
        for frame in self.frame_generator:
            log_progress(logger, "Processing frames for crop extraction", frame_idx, total_frames)

            detections = all_detections_dq.popleft()

            # Skip if no detections
            if len(detections) == 0:
                frame_idx += 1
                continue
            
            # Only extract crops every Nth frame (configurable)
            should_extract_crops = (frame_idx % self.crop_extract_interval) == 0
            
            if not should_extract_crops:
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
                # Validate bounding box coordinates
                frame_height, frame_width = frame.shape[:2]
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(x1 + 1, min(x2, frame_width))
                y2 = max(y1 + 1, min(y2, frame_height))
                # Extract crop and validate it's not empty
                crop = frame[y1:y2, x1:x2]
                # Skip if crop is empty or too small
                if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                    logger.warning(f"Skipping invalid crop for tracker {tracker_id} at frame {frame_id}: bbox=({x1},{y1},{x2},{y2}), crop_shape={crop.shape}")
                    continue
                # Track crop size (area in pixels), width, and height
                crop_height, crop_width = crop.shape[0], crop.shape[1]
                crop_sizes.append(crop_height * crop_width)
                crop_widths.append(crop_width)
                crop_heights.append(crop_height)
                # Ensure the directory for this tracker_id exists
                tracker_dir = os.path.join(all_crops_dir, str(tracker_id))
                os.makedirs(tracker_dir, exist_ok=True)
                # Save crop with filename: frame_id_tracker_id_confidence.jpg
                crop_filename = f"{frame_id}_{tracker_id}_{confidence:.3f}.jpg"
                crop_path = os.path.join(tracker_dir, crop_filename)
                # Validate crop before writing
                if crop is not None and crop.size > 0:
                    success = cv2.imwrite(crop_path, crop)
                    if success:
                        total_crops_extracted += 1
                    else:
                        logger.warning(f"Failed to write crop to {crop_path}")
                else:
                    logger.warning(f"Invalid crop for tracker {tracker_id} at frame {frame_id}")
                    continue
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

        if total_crops_extracted == 0:
            logger.critical("")
            raise RuntimeError("No crops were extracted. Check your detections and video frames.")

        # Log crop size and dimension statistics
        if crop_sizes:
            min_width = int(np.min(crop_widths))
            max_width = int(np.max(crop_widths))
            median_width = int(np.median(crop_widths))
            min_height = int(np.min(crop_heights))
            max_height = int(np.max(crop_heights))
            median_height = int(np.median(crop_heights))
            min_size = int(np.min(crop_sizes))
            max_size = int(np.max(crop_sizes))
            median_size = int(np.median(crop_sizes))
            logger.info(f"Crop width range: {min_width} - {max_width} (median: {median_width})")
            logger.info(f"Crop height range: {min_height} - {max_height} (median: {median_height})")
            logger.info(f"Crop size (pixels): {min_size} - {max_size} (median: {median_size})")

        logger.info(f"Crop extraction complete! {total_crops_extracted} crops extracted.")

    def get_crops_directory (self) -> str:
        return self.crops_dir
    
    def get_all_crops_directory(self) -> str:
        """
        Returns the directory where all crops are stored.
        """
        return self.all_crops_dir

    def reorganize_crops_by_stitched_tracks(self, track_id_mapping: dict[int, int]) -> None:
        """
        Reorganizes the all_crops_directory based on stitched tracks mapping.
        
        This function moves crops from original track directories to new stitched track directories
        based on the mapping provided by the track stitching algorithm.
        
        Args:
            track_id_mapping: Dictionary mapping original track IDs to stitched track IDs
        """
        logger.info(f"Reorganizing crops based on stitched tracks mapping")
        logger.info(f"Processing {len(track_id_mapping)} track mappings")
        
        if not hasattr(self, 'all_crops_dir') or not os.path.exists(self.all_crops_dir):
            logger.error("all_crops_dir not found. Make sure extract_crops() was called first.")
            return
            
        # Create a temporary directory for reorganization
        temp_reorganize_dir = os.path.join(self.temp_dir, "temp_reorganize")
        os.makedirs(temp_reorganize_dir, exist_ok=True)
        
        # Create directories for each stitched track ID
        stitched_track_ids = set(track_id_mapping.values())
        stitched_dirs = {}
        
        for stitched_track_id in stitched_track_ids:
            stitched_dir = os.path.join(temp_reorganize_dir, str(stitched_track_id))
            os.makedirs(stitched_dir, exist_ok=True)
            stitched_dirs[stitched_track_id] = stitched_dir
        
        # Move crops from original track directories to stitched track directories
        total_crops_moved = 0
        processed_original_tracks = 0
        
        for original_track_id, stitched_track_id in track_id_mapping.items():
            original_track_dir = os.path.join(self.all_crops_dir, str(original_track_id))
            
            if not os.path.exists(original_track_dir):
                logger.warning(f"Original track directory not found: {original_track_dir}")
                continue
                
            processed_original_tracks += 1
            stitched_dir = stitched_dirs[stitched_track_id]
            
            # Move all crops from original track directory to stitched track directory
            crop_files = [f for f in os.listdir(original_track_dir) if f.endswith('.jpg')]
            
            for crop_file in crop_files:
                src_path = os.path.join(original_track_dir, crop_file)
                dst_path = os.path.join(stitched_dir, crop_file)
                
                # Handle potential filename conflicts by adding a suffix
                counter = 1
                base_name, ext = os.path.splitext(crop_file)
                while os.path.exists(dst_path):
                    new_name = f"{base_name}_{counter}{ext}"
                    dst_path = os.path.join(stitched_dir, new_name)
                    counter += 1
                
                try:
                    shutil.move(src_path, dst_path)
                    total_crops_moved += 1
                except Exception as e:
                    logger.error(f"Failed to move crop {src_path} to {dst_path}: {e}")
            
            # Remove the now-empty original track directory
            try:
                os.rmdir(original_track_dir)
            except OSError as e:
                logger.warning(f"Failed to remove empty directory {original_track_dir}: {e}")
        
        # Replace the original all_crops_dir with the reorganized one
        original_all_crops_backup = os.path.join(self.temp_dir, "all_crops_backup")
        
        try:
            # Backup original directory
            if os.path.exists(original_all_crops_backup):
                shutil.rmtree(original_all_crops_backup)
            shutil.move(self.all_crops_dir, original_all_crops_backup)
            
            # Move reorganized directory to replace original
            shutil.move(temp_reorganize_dir, self.all_crops_dir)
            
            # Remove backup after successful reorganization
            shutil.rmtree(original_all_crops_backup)
            
        except Exception as e:
            logger.error(f"Failed to replace all_crops_dir: {e}")
            # Restore from backup if something went wrong
            if os.path.exists(original_all_crops_backup):
                if os.path.exists(self.all_crops_dir):
                    shutil.rmtree(self.all_crops_dir)
                shutil.move(original_all_crops_backup, self.all_crops_dir)
                logger.info("Restored original all_crops_dir from backup")
            return
        
        logger.info(f"Crop reorganization complete!")
        logger.info(f"Processed {processed_original_tracks} original track directories")
        logger.info(f"Moved {total_crops_moved} crops to {len(stitched_track_ids)} stitched track directories")
        logger.info(f"Track count reduced from {len(track_id_mapping)} to {len(stitched_track_ids)}")

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


