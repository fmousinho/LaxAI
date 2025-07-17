"""
Crop extraction functionality for extracting player crops from video frames.

This module provides functional approaches to crop extraction, reorganization, and train/val splitting.
"""

import os
import cv2
import shutil
import random
import logging
import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Any

import supervision as sv

from config.all_config import detection_config
from config.transforms import ensure_rgb_format
from modules.utils import log_progress

logger = logging.getLogger(__name__)


def extract_crops_from_video(
    frame_generator: Any,
    all_detections: List[sv.Detections],
    temp_dir: str,
    crop_extract_interval: int = detection_config.crop_extract_interval
) -> Tuple[str, str]:
    """
    Extracts crops from video frames based on detections and organizes them by track_id.
    
    Args:
        frame_generator: Generator yielding video frames
        all_detections: List of detections for each frame
        temp_dir: Temporary directory for storing crops
        crop_extract_interval: Extract crops every N frames
        
    Returns:
        Tuple of (crops_dir, all_crops_dir) paths
        
    Raises:
        RuntimeError: If no crops were extracted
    """
    # Setup directories
    crops_dir = os.path.join(temp_dir, "crops")
    all_crops_dir = os.path.join(crops_dir, "all_crops")
    os.makedirs(all_crops_dir, exist_ok=True)

    all_detections_copy = all_detections.copy()
    
    logger.info(f"Starting crop extraction.")
    logger.info(f"directory: {all_crops_dir}")
    
    frame_idx = 0
    all_detections_dq = deque(all_detections_copy)
    
    # Calculate total frames for progress logging
    total_frames = len(all_detections_copy) if all_detections_copy else 0
    total_crops_extracted = 0
    
    # Statistics tracking
    crop_sizes = []
    crop_widths = []
    crop_heights = []
    
    for frame in frame_generator:
        log_progress(logger, "Processing frames for crop extraction", frame_idx, total_frames)
        
        if not all_detections_dq:
            break
            
        detections = all_detections_dq.popleft()
        
        # Skip if no detections
        if len(detections) == 0:
            frame_idx += 1
            continue
        
        # Only extract crops every Nth frame (configurable)
        should_extract_crops = (frame_idx % crop_extract_interval) == 0
        
        if not should_extract_crops:
            frame_idx += 1
            continue
        
        # Extract crops for each detection in this frame
        crops_extracted_this_frame = _extract_crops_from_frame(
            frame, detections, frame_idx, all_crops_dir, 
            crop_sizes, crop_widths, crop_heights
        )
        total_crops_extracted += crops_extracted_this_frame
        
        frame_idx += 1
    
    if total_crops_extracted == 0:
        logger.critical("No crops were extracted. Check your detections and video frames.")
        raise RuntimeError("No crops were extracted. Check your detections and video frames.")
    
    # Log statistics
    _log_crop_statistics(crop_sizes, crop_widths, crop_heights)
    
    logger.info(f"Crop extraction complete! {total_crops_extracted} crops extracted.")
    
    return crops_dir, all_crops_dir


def _extract_crops_from_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    frame_idx: int,
    all_crops_dir: str,
    crop_sizes: List[int],
    crop_widths: List[int],
    crop_heights: List[int]
) -> int:
    """
    Extract crops from a single frame based on detections.
    
    Returns:
        Number of crops extracted from this frame
    """
    crops_extracted = 0
    
    for i in range(len(detections)):
        # Get detection data
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
        
        # Extract and validate crop
        crop_rgb = _extract_and_validate_crop(frame, bbox)
        if crop_rgb is None:
            logger.warning(f"Skipping invalid crop for tracker {tracker_id} at frame {frame_id}")
            continue
        
        # Track crop statistics
        crop_height, crop_width = crop_rgb.shape[0], crop_rgb.shape[1]
        crop_sizes.append(crop_height * crop_width)
        crop_widths.append(crop_width)
        crop_heights.append(crop_height)
        
        # Save crop
        if _save_crop(crop_rgb, tracker_id, frame_id, confidence, all_crops_dir):
            crops_extracted += 1
    
    return crops_extracted


def _extract_and_validate_crop(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract and validate a crop from a frame.
    
    Returns:
        Crop as RGB numpy array, or None if invalid
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Validate bounding box coordinates
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))
    
    # Extract crop and convert to RGB
    crop_bgr = frame[y1:y2, x1:x2]
    crop_rgb = ensure_rgb_format(crop_bgr, "BGR")
    
    # Validate crop size
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 5 or crop_rgb.shape[1] < 5:
        return None
    
    return crop_rgb


def _save_crop(
    crop_rgb: np.ndarray,
    tracker_id: int,
    frame_id: int,
    confidence: float,
    all_crops_dir: str
) -> bool:
    """
    Save a crop to the appropriate directory.
    
    Returns:
        True if successful, False otherwise
    """
    # Ensure the directory for this tracker_id exists
    tracker_dir = os.path.join(all_crops_dir, str(tracker_id))
    os.makedirs(tracker_dir, exist_ok=True)
    
    # Save crop with filename: frame_id_tracker_id_confidence.jpg
    crop_filename = f"{frame_id}_{tracker_id}_{confidence:.3f}.jpg"
    crop_path = os.path.join(tracker_dir, crop_filename)
    
    # Validate crop before writing
    if crop_rgb is not None and crop_rgb.size > 0:
        success = cv2.imwrite(crop_path, crop_rgb)
        if not success:
            logger.warning(f"Failed to write crop to {crop_path}")
        return success
    else:
        logger.warning(f"Invalid crop for tracker {tracker_id} at frame {frame_id}")
        return False


def _log_crop_statistics(crop_sizes: List[int], crop_widths: List[int], crop_heights: List[int]) -> None:
    """Log crop size and dimension statistics."""
    if not crop_sizes:
        return
        
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


def reorganize_crops_by_stitched_tracks(
    all_crops_dir: str,
    temp_dir: str,
    track_id_mapping: dict[int, int]
) -> None:
    """
    Reorganizes the all_crops_directory based on stitched tracks mapping.
    
    This function moves crops from original track directories to new stitched track directories
    based on the mapping provided by the track stitching algorithm.
    
    Args:
        all_crops_dir: Directory containing the original track crops
        temp_dir: Temporary directory for reorganization
        track_id_mapping: Dictionary mapping original track IDs to stitched track IDs
    """
    logger.info(f"Reorganizing crops based on stitched tracks mapping")
    logger.info(f"Processing {len(track_id_mapping)} track mappings")
    
    if not os.path.exists(all_crops_dir):
        logger.error(f"all_crops_dir not found: {all_crops_dir}")
        return
    
    # Create a temporary directory for reorganization
    temp_reorganize_dir = os.path.join(temp_dir, "temp_reorganize")
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
        original_track_dir = os.path.join(all_crops_dir, str(original_track_id))
        
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
    _replace_directory_atomically(all_crops_dir, temp_reorganize_dir, temp_dir)
    
    logger.info(f"Crop reorganization complete!")
    logger.info(f"Processed {processed_original_tracks} original track directories")
    logger.info(f"Moved {total_crops_moved} crops to {len(stitched_track_ids)} stitched track directories")
    logger.info(f"Track count reduced from {len(track_id_mapping)} to {len(stitched_track_ids)}")


def _replace_directory_atomically(target_dir: str, source_dir: str, temp_dir: str) -> None:
    """
    Atomically replace target_dir with source_dir, with backup for rollback.
    """
    backup_dir = os.path.join(temp_dir, "all_crops_backup")
    
    try:
        # Backup original directory
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(target_dir, backup_dir)
        
        # Move reorganized directory to replace original
        shutil.move(source_dir, target_dir)
        
        # Remove backup after successful reorganization
        shutil.rmtree(backup_dir)
        
    except Exception as e:
        logger.error(f"Failed to replace directory {target_dir}: {e}")
        # Restore from backup if something went wrong
        if os.path.exists(backup_dir):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(backup_dir, target_dir)
            logger.info("Restored original directory from backup")


def create_train_val_split(
    source_folder: str,
    destin_folder: str,
    train_ratio: float = 0.8
) -> None:
    """
    Splits the extracted crops into training and validation sets, maintaining per-track structure.
    
    Args:
        source_folder: Directory containing the source crops folders (each folder contains images)
        destin_folder: Directory where train and val folders will be created
        train_ratio: Ratio of data to use for training (default: 0.8)
        
    Raises:
        FileNotFoundError: If source folder doesn't exist
    """
    logger.info(f"Creating train/val split:")
    logger.info(f" source: {source_folder}")
    logger.info(f" destination: {destin_folder}")
    
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


# Backward compatibility wrapper class (optional - can be removed later)
class CropExtractor:
    """
    Backward compatibility wrapper for the functional crop extraction.
    
    This class provides the same interface as the original CropExtractor class
    but uses the new functional implementation internally.
    """
    
    def __init__(self, frame_generator, all_detections: List[sv.Detections], temp_dir: str, 
                 crop_extract_interval: int = detection_config.crop_extract_interval):
        self.frame_generator = frame_generator
        self.all_detections = all_detections
        self.temp_dir = temp_dir
        self.crop_extract_interval = crop_extract_interval
        self.crops_dir: Optional[str] = None
        self.all_crops_dir: Optional[str] = None
    
    def extract_crops(self) -> None:
        """Extract crops using the functional implementation."""
        self.crops_dir, self.all_crops_dir = extract_crops_from_video(
            self.frame_generator,
            self.all_detections,
            self.temp_dir,
            self.crop_extract_interval
        )
    
    def get_crops_directory(self) -> str:
        """Get the crops directory path."""
        if self.crops_dir is None:
            raise RuntimeError("extract_crops() must be called first")
        return self.crops_dir
    
    def get_all_crops_directory(self) -> str:
        """Get the all crops directory path."""
        if self.all_crops_dir is None:
            raise RuntimeError("extract_crops() must be called first")
        return self.all_crops_dir
    
    def reorganize_crops_by_stitched_tracks(self, track_id_mapping: dict[int, int]) -> None:
        """Reorganize crops using the functional implementation."""
        if self.all_crops_dir is None:
            raise RuntimeError("extract_crops() must be called first")
        reorganize_crops_by_stitched_tracks(self.all_crops_dir, self.temp_dir, track_id_mapping)
