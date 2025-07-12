import os
import cv2
from collections import deque
import shutil
import random
import supervision as sv
import logging
import numpy as np
from itertools import tee
from .utils import log_progress

logger = logging.getLogger(__name__)

from config.all_config import detection_config
from config.transforms import ensure_rgb_format

class CropExtractor:
    """
    Extracts crops from video frames based on detections and organizes them by track_id.
    Each crop is saved in a directory named after the track_id.
    """

    def __init__(self, frame_generator, all_detections: list[sv.Detections], temp_dir: str, crop_extract_interval: int = detection_config.crop_extract_interval, grass_mask: bool = detection_config.grass_mask):
        self.all_detections = all_detections
        self.temp_dir = temp_dir
        self.crop_extract_interval = crop_extract_interval
        self.grass_masker = None  # Initialize grass_masker
        
        if grass_mask:
            self.grass_mask = grass_mask
            self.frame_generator, grass_generator = tee(self.frame_generator, 2)
            self.mask_this = self._create_grass_masker(grass_generator)
        else:
            self.frame_generator = frame_generator



    def _create_grass_masker(self, frame_generator):
        """
        Creates a grass masking function by analyzing frame samples to determine grass color ranges.
        
        Analyzes the top 50% and bottom 10% of 5 equally separated frames to determine grass color
        statistics, then returns a callable that masks grass-colored pixels to white.
        
        Args:
            frame_generator: Generator yielding video frames
            
        Returns:
            Callable that takes a crop and returns the same crop with grass pixels turned white
        """
        logger.info("Creating grass masker by analyzing frame samples...")
        
        # Convert generator to list to allow indexing
        frames = list(frame_generator)
        total_frames = len(frames)
        
        if total_frames < 5:
            logger.warning(f"Not enough frames ({total_frames}) for grass analysis. Using first frame only.")
            sample_indices = [0]
        else:
            # Select 5 equally separated frames
            sample_indices = [int(i * total_frames / 5) for i in range(5)]
        
        grass_colors = []
        
        for idx in sample_indices:
            frame = frames[idx]
            frame_height, frame_width = frame.shape[:2]
            
            # Extract top 50% and bottom 10% regions
            top_region = frame[:frame_height//2, :]  # Top 50%
            bottom_start = int(frame_height * 0.9)
            bottom_region = frame[bottom_start:, :]  # Bottom 10%
            
            # Combine regions for grass analysis
            grass_regions = [top_region, bottom_region]
            
            for region in grass_regions:
                if region.size > 0:
                    # Convert BGR to RGB first, then to HSV
                    region_rgb = ensure_rgb_format(region, "BGR")
                    region_hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
                    
                    # Reshape to get all pixels as rows
                    pixels_hsv = region_hsv.reshape(-1, 3)
                    
                    # Filter for green-ish hues (grass colors)
                    # HSV hue range: 0-179, typical grass is around 40-80 (green spectrum)
                    hue = pixels_hsv[:, 0]
                    saturation = pixels_hsv[:, 1]
                    value = pixels_hsv[:, 2]
                    
                    # Filter for grass-like colors:
                    # - Hue in green range (30-90)
                    # - Moderate to high saturation (40-255) to avoid grays
                    # - Moderate brightness (30-200) to avoid shadows and highlights
                    grass_mask = (
                        (hue >= 30) & (hue <= 90) &          # Green hue range
                        (saturation >= 40) & (saturation <= 255) &  # Avoid desaturated colors
                        (value >= 30) & (value <= 200)       # Avoid very dark/bright
                    )
                    
                    filtered_pixels_hsv = pixels_hsv[grass_mask]
                    
                    if len(filtered_pixels_hsv) > 0:
                        grass_colors.append(filtered_pixels_hsv)
        
        if not grass_colors:
            logger.warning("No grass colors found in frame analysis. Masking will be disabled.")
            return lambda crop: crop
        
        # Combine all grass color samples
        all_grass_pixels_hsv = np.vstack(grass_colors)
        
        # Calculate mean and standard deviation for each HSV channel
        grass_mean_hsv = np.mean(all_grass_pixels_hsv, axis=0)
        grass_std_hsv = np.std(all_grass_pixels_hsv, axis=0)
        
        # Define bounds: mean ± 1 standard deviation
        lower_bound_hsv = grass_mean_hsv - grass_std_hsv
        upper_bound_hsv = grass_mean_hsv + grass_std_hsv
        
        # Ensure bounds are within valid HSV range
        # H: 0-179, S: 0-255, V: 0-255
        lower_bound_hsv = np.clip(lower_bound_hsv, [0, 0, 0], [179, 255, 255])
        upper_bound_hsv = np.clip(upper_bound_hsv, [0, 0, 0], [179, 255, 255])
        
        # Convert mean HSV back to RGB for logging
        mean_hsv_reshaped = grass_mean_hsv.reshape(1, 1, 3).astype(np.uint8)
        mean_rgb = cv2.cvtColor(mean_hsv_reshaped, cv2.COLOR_HSV2RGB)[0, 0]
        
        logger.info(f"Grass color analysis complete (HSV-based):")
        logger.info(f"  Mean HSV: ({grass_mean_hsv[0]:.1f}, {grass_mean_hsv[1]:.1f}, {grass_mean_hsv[2]:.1f})")
        logger.info(f"  Mean RGB equivalent: ({mean_rgb[0]}, {mean_rgb[1]}, {mean_rgb[2]})")
        logger.info(f"  Std HSV: ({grass_std_hsv[0]:.1f}, {grass_std_hsv[1]:.1f}, {grass_std_hsv[2]:.1f})")
        logger.info(f"  HSV Lower bound: ({lower_bound_hsv[0]:.1f}, {lower_bound_hsv[1]:.1f}, {lower_bound_hsv[2]:.1f})")
        logger.info(f"  HSV Upper bound: ({upper_bound_hsv[0]:.1f}, {upper_bound_hsv[1]:.1f}, {upper_bound_hsv[2]:.1f})")
        
        def grass_mask_function(crop):
            """
            Masks grass-colored pixels in a crop by turning them white.
            
            Args:
                crop: Input crop as numpy array (assumed to be RGB)
                
            Returns:
                Crop with grass pixels turned white
            """
            if crop.size == 0:
                return crop
            
            # Ensure crop is in RGB format
            crop_rgb = ensure_rgb_format(crop, "RGB")  # Assume input is already RGB
            
            # Convert crop to HSV for color-based masking
            crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
            
            # Create mask for grass colors in HSV space
            # Check if each pixel falls within the grass HSV bounds
            mask = np.all((crop_hsv >= lower_bound_hsv) & (crop_hsv <= upper_bound_hsv), axis=2)
            
            # Create output crop (copy to avoid modifying original)
            masked_crop = crop_rgb.copy()
            
            # Set grass pixels to white (255, 255, 255)
            masked_crop[mask] = [255, 255, 255]
            
            return masked_crop
        
        return grass_mask_function


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
                crop_bgr = frame[y1:y2, x1:x2]
                crop_rgb = ensure_rgb_format(crop_bgr, "BGR")
                
                # Apply grass masking if enabled
                if hasattr(self, 'grass_masker') and self.grass_masker is not None:
                    crop_rgb = self.grass_masker(crop_rgb)
                
                # Skip if crop is empty or too small
                if crop_rgb.size == 0 or crop_rgb.shape[0] < 5 or crop_rgb.shape[1] < 5:
                    logger.warning(f"Skipping invalid crop for tracker {tracker_id} at frame {frame_id}: bbox=({x1},{y1},{x2},{y2}), crop_shape={crop_rgb.shape}")
                    continue
                # Track crop size (area in pixels), width, and height
                crop_height, crop_width = crop_rgb.shape[0], crop_rgb.shape[1]
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
                if crop_rgb is not None and crop_rgb.size > 0:
                    success = cv2.imwrite(crop_path, crop_rgb)
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
    logger.info(f"Creating train/val split:")
    logger.info(f" source: {source_folder}")
    logger.info(f" destination:to {destin_folder}")

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



