"""
Data preparation pipeline for LaxAI project.

This module defines the DataPrepPipeline class and related utilities for processing raw video data,
including downloading from Google Storage, extracting frames, augmenting images, and preparing datasets for training.
"""


import os
import sys
import json
import uuid
import logging
import time
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import cv2
import numpy as np
import supervision as sv
from supervision import Detections

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.common.google_storage import get_storage
from core.common.detection import DetectionModel
from core.common.pipeline_step import PipelineStep, StepStatus
from core.common.pipeline import Pipeline, PipelineStatus
from core.config.all_config import DetectionConfig
from core.config import logging_config
from core.common.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
from core.common.crop_utils import extract_crops_from_video, create_train_val_split
from core.train.augmentation import augment_images

logger = logging.getLogger(__name__)


class DataPrepPipeline(Pipeline):
    """
    Data preparation pipeline that processes MP4 videos from Google Storage.

    This pipeline:
    1. Creates a processing run folder with GUID
    2. Processes videos from the raw directory
    3. Validates video resolution (minimum 1920 by 1080)
    4. Extracts frames with sufficient detections
    5. Saves metadata and detection results
    """
    
    def __init__(self, config: DetectionConfig, tenant_id: str = "tenant1", verbose: bool = True, save_intermediate: bool = True):
        """
        Initialize the training pipeline.
        
        Args:
            config: Detection configuration object
            tenant_id: The tenant ID to process videos for (default: "tenant1")
            verbose: Enable verbose logging (default: False)
            save_intermediate: Save intermediate results for each step (default: False)
        """
        self.config = config
        self.tenant_id = tenant_id
        self.frames_per_video = config.frames_per_video
        self.background_mask_detector = BackgroundMaskDetector()
        
        # Get storage clients
        self.storage_admin = get_storage("common")  # For accessing common resources
        self.tenant_storage = get_storage(f"{tenant_id}/user")  # For tenant-specific operations
        
        # Detection model is required for training pipeline
        try:
            self.detection_model = DetectionModel()
            logger.info("Detection model successfully loaded")
        except RuntimeError as e:
            logger.critical(f"CRITICAL ERROR: Detection model is required for training pipeline but failed to load: {e}")
            raise RuntimeError(f"Training pipeline cannot continue without detection model: {e}")
        
        # Define pipeline steps
        step_definitions = {
            "import_videos": {
                "description": "Move MP4 videos from tenant's raw directory for processing",
                "function": self._import_video
            },
            "load_videos": {
                "description": "Download videos to memory",
                "function": self._load_video
            },
            "extract_frames": {
                "description": "Extract frames with sufficient detections for processing",
                "function": self._extract_frames_for_detections
            },
            "calculate_grass_mask": {
                "description": "Calculate grass mask for each frame",
                "function": self._initialize_grass_mask
            },
            "detect_players": {
                "description": "Process detection results and save to storage",
                "function": self._detect_players
            },
            "extract_crops": {
                "description": "Extract and save player crops from detections",
                "function": self._extract_crops
            },
            "remove_crop_background": {
                "description": "Remove background from player crops",
                "function": self._remove_crop_background
            },
            "augment_crops": {
                "description": "Augment player crops for training",
                "function": self._augment_crops
            },
            "create_training_and_validation_sets": {
                "description": "Create training and validation datasets from processed crops",
                "function": self._create_training_and_validation_sets
            }
        }

        # Initialize base pipeline
        super().__init__(
            pipeline_name="train_pipeline",
            storage_client=self.tenant_storage,
            step_definitions=step_definitions,
            verbose=verbose,
            save_intermediate=save_intermediate
        )
    
    def run(self, video_path: str, resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Execute the complete training pipeline for a single video.
        
        Args:
            video_path: Path to the video file to process (can be local path or gs:// URL)
            resume_from_checkpoint: Whether to check for and resume from existing checkpoint (default: True)
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        if not video_path:
            return {"status": PipelineStatus.ERROR.value, "error": "No video path provided"}

        logger.info(f"Processing video: {video_path}")

        # Create initial context with the video path
        initial_context = {"video_path": video_path}
        
        # Call the base class run method with the initial context
        # The base class now handles checkpoint functionality automatically
        results = super().run(initial_context, resume_from_checkpoint=resume_from_checkpoint)
        
        # Add training-specific result formatting
        context = results.get("context", {})
        
        # Format results to match expected structure
        formatted_results = {
            "status": results["status"],
            "run_guid": results["run_guid"],
            "run_folder": results["run_folder"],
            "video_path": video_path,
            "video_guid": context.get("video_guid", "unknown"),
            "video_folder": context.get("video_folder", "unknown"),
            "errors": results["errors"],
            "pipeline_summary": results["pipeline_summary"]
        }
        
        return formatted_results
    
    def _import_video(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a single video from the provided path into organized video folder.
        
        Args:
            context: Pipeline context containing video_path
            
        Returns:
            Dictionary with imported video information
        """
        video_path = context.get("video_path")
        if not video_path:
            return {"status": StepStatus.ERROR.value, "error": "No video path provided"}

        try:
            logger.info(f"Importing video: {video_path}")

            # Generate video GUID and folder
            video_guid = str(uuid.uuid4())
            video_folder = f"{self.run_folder}/video_{video_guid}"

            # Check if it's a GCS URL, local file, or blob path
            if video_path.startswith("gs://"):
                imported_video = self._import_single_video(video_path, video_guid, video_folder)
            elif "/" in video_path:
                # It's a blob path (contains slashes), treat as Google Storage blob
                logger.info(f"Processing video blob: {video_path}")
                imported_video = self._import_single_video(video_path, video_guid, video_folder)
            else:
                # It's just a filename, assume it's in the tenant's raw directory
                raw_video_path = f"raw/{video_path}"
                logger.info(f"Video filename provided, looking in tenant raw directory: {raw_video_path}")
                imported_video = self._import_single_video(raw_video_path, video_guid, video_folder)

            # Check if the import succeeded
            if not imported_video:
                raise ValueError(f"Error importing video")

            logger.info(f"Successfully imported {imported_video}")

            return {"imported_video": imported_video, "video_guid": video_guid, "video_folder": video_folder}

        except Exception as e:
            logger.error(f"Failed to import video {video_path}: {str(e)}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    
    def _import_single_video(self, video_blob_name: str, video_guid: str, video_folder: str) -> str:
        """
        Import a single video from raw directory into organized video folder.
        
        Args:
            video_blob_name: The blob name of the video in Google Storage
            video_guid: The GUID for this video
            video_folder: The folder where the video should be stored
            
        Returns:
            Path to the imported video file
            
        Raises:
            RuntimeError: If import operation fails
        """
        logger.info(f"Importing video: {video_blob_name}")
        
        # Create a robust temporary file path
        temp_video_path = f"/tmp/video_{video_guid}.mp4"
        
        try:
            # Move video to its new organized folder with GUID name
            new_video_name = f"{video_guid}.mp4"
            new_video_blob = f"{video_folder}/{new_video_name}"
            
            if not self.tenant_storage.move_blob(video_blob_name, new_video_blob):
                raise RuntimeError(f"Failed to move video to new location: {video_blob_name} -> {new_video_blob}")
            
            # Download the moved video file for metadata extraction
            if not self.tenant_storage.download_blob(new_video_blob, temp_video_path):
                raise RuntimeError(f"Failed to download moved video: {new_video_blob}")
            
            # Extract video metadata from the temporary file
            metadata = self._extract_video_metadata(temp_video_path, video_blob_name, video_guid)
            
            # Save metadata file
            metadata_blob = f"{video_folder}/metadata.json"
            metadata_content = json.dumps(metadata, indent=2)
            if not self.tenant_storage.upload_from_string(metadata_blob, metadata_content):
                raise RuntimeError(f"Failed to save metadata for video: {video_blob_name}")
            
            logger.info(f"Successfully imported video: {video_blob_name} -> {new_video_blob}")
            return new_video_blob
            
        except Exception as e:
            raise RuntimeError(f"Failed to import video {video_blob_name}: {str(e)}")
        finally:
            # Always clean up temporary file, even if there's an error
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    logger.debug(f"Cleaned up temporary file: {temp_video_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file {temp_video_path}: {cleanup_error}")
    
    def _load_video(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load the imported video into memory for processing.
        
        Args:
            context: Pipeline context containing imported_video
            
        Returns:
            Dictionary with loaded video information
        """
        imported_video = context.get("imported_video")
        video_guid = context.get("video_guid")
        
        if not imported_video:
            logger.error("No imported video to load")
            return {"status": StepStatus.ERROR.value, "error": "No imported video provided"}
        
        # Create a robust temporary file path
        temp_video_path = f"/tmp/processing_{video_guid}.mp4"
        
        try:
            # Always download from Google Storage as we're now blob-based
            logger.info(f"Downloading video from storage: {imported_video}")
            if not self.tenant_storage.download_blob(imported_video, temp_video_path):
                raise RuntimeError(f"Failed to download video for processing: {imported_video}")
            
            # Validate video resolution
            logger.info(f"Validating video resolution: {temp_video_path}")
            if not self._validate_video_resolution(temp_video_path):
                raise RuntimeError(f"Video resolution validation failed for: {imported_video}. Video must be at least 1920x1080 pixels.")
            
            logger.info(f"Successfully loaded video for processing: {imported_video}")
            
            return {
                "loaded_video": {
                    "video_path": imported_video,
                    "video_guid": video_guid,
                    "temp_path": temp_video_path
                }
            }
            
        except Exception as e:
            logger.error(f"Critical error loading video {imported_video}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
        finally:
            # Note: We don't cleanup the temp file here because it's needed for subsequent steps
            # The cleanup will happen in the pipeline's cleanup method or after processing
            pass
    
    def _detect_players(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run player detection on the extracted frames.
        
        Args:
            context: Pipeline context containing frames_data
            
        Returns:
            Dictionary with detection results
        """
        frames_data = context.get("frames_data")
        video_guid = context.get("video_guid")
        
        if not frames_data:
            # If frames_data is missing (e.g., when resuming from checkpoint), try to regenerate it
            logger.warning("No frames data for player detection - attempting to regenerate from video")
            
            # Try to get loaded_video info to re-extract frames
            loaded_video = context.get("loaded_video")
            if loaded_video and loaded_video.get("temp_path"):
                logger.info("Attempting to re-extract frames for detection")
                # Re-extract frames for this step
                frames_result = self._extract_frames_for_detections(context)
                if frames_result.get("status") == StepStatus.ERROR.value:
                    return frames_result
                frames_data = frames_result.get("frames_data")
                # Update context with re-extracted frames
                context["frames_data"] = frames_data
            
            if not frames_data:
                logger.error("No frames data for player detection - previous step failed or frames unavailable")
                return {"status": StepStatus.ERROR.value, "error": "No frames data provided - frame extraction failed"}
        
        if not video_guid:
            logger.error("No video GUID for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No video GUID provided"}
        
        try:
            logger.info(f"Starting player detection for video: {video_guid} with {len(frames_data)} frames")
            
            # Process each frame individually since the detection model expects individual frames
            all_detections = []
            for i, frame in enumerate(frames_data):
                try:
                    # Ensure frame is a numpy array with correct format
                    if not isinstance(frame, np.ndarray):
                        logger.warning(f"Frame {i} is not a numpy array, skipping")
                        continue
                    
                    # Convert from BGR to RGB if needed (OpenCV uses BGR, models usually expect RGB)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame
                    
                    # Generate detections for this single frame
                    frame_detections = self.detection_model.generate_detections(frame_rgb)
                    
                    if frame_detections is not None and len(frame_detections) > 0:
                        # Append the entire Detections object to the list
                        all_detections.append(frame_detections)
                        logger.debug(f"Frame {i}: Found {len(frame_detections)} detections")
                    else:
                        logger.debug(f"Frame {i}: No detections found")
                        
                except Exception as frame_error:
                    logger.warning(f"Failed to process frame {i}: {frame_error}")
                    continue

            if len(all_detections) == 0:
                logger.error(f"Could not extract detections from {len(frames_data)} frames for video {video_guid}")
                return {"status": StepStatus.ERROR.value, "error": f"No detections found in {len(frames_data)} frames"}

            # Save detections - convert to JSON-serializable format
            video_folder = context.get("video_folder", f"{self.run_folder}/video_{video_guid}")
            detections_blob = f"{video_folder}/detections.json"

            # Convert detections to JSON-serializable format
            serializable_detections = []
            for frame_idx, detection_obj in enumerate(all_detections):
                # Each detection_obj is a supervision.Detections object
                if hasattr(detection_obj, 'xyxy') and hasattr(detection_obj, 'confidence') and hasattr(detection_obj, 'class_id'):
                    # Convert supervision.Detections to dictionary with frame information
                    frame_detections = {
                        "frame_index": frame_idx,
                        "boxes": detection_obj.xyxy.tolist() if hasattr(detection_obj.xyxy, 'tolist') else detection_obj.xyxy,
                        "confidence": detection_obj.confidence.tolist() if hasattr(detection_obj.confidence, 'tolist') else detection_obj.confidence,
                        "class_id": detection_obj.class_id.tolist() if hasattr(detection_obj.class_id, 'tolist') else detection_obj.class_id,
                        "tracker_id": detection_obj.tracker_id.tolist() if detection_obj.tracker_id is not None and hasattr(detection_obj.tracker_id, 'tolist') else detection_obj.tracker_id,
                        "num_detections": len(detection_obj)
                    }
                    serializable_detections.append(frame_detections)
                else:
                    # If it's already a dict or other serializable format, keep it as is
                    serializable_detections.append(detection_obj)

            detections_content = json.dumps(serializable_detections, indent=2)
            if not self.tenant_storage.upload_from_string(detections_blob, detections_content):
                logger.error(f"Failed to save detections for video: {video_guid}")
                return {"status": StepStatus.ERROR.value, "error": f"Failed to save detections to storage: {detections_blob}"}
            
            logger.info(f"Completed detection for video: {video_guid} - found {len(all_detections)} detections")
            
            return {
                "detection_result": {
                    "video_guid": video_guid,
                    "frames_data": frames_data,
                    "detections_blob": detections_blob
                }
            }
            
        except Exception as e:
            logger.error(f"Critical error in player detection for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    
    def _extract_crops(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and save player crops from detections.
        
        Args:
            context: Pipeline context containing detection_result and frames_data
            
        Returns:
            Dictionary with crop extraction results
        """
        detection_result = context.get("detection_result")
        frames_data = context.get("frames_data")
        video_guid = context.get("video_guid")
        
        if not detection_result or not frames_data:
            logger.warning("No detection result or frames data found for crop extraction")
            return {"status": StepStatus.ERROR.value, "error": "No detection result or frames data provided"}
        
        try:
            logger.info(f"Extracting crops for video: {video_guid}")
            
            # Get the video folder
            video_folder = context.get("video_folder", f"{self.run_folder}/video_{video_guid}")
            
            # Create crops directory structure
            crops_folder = f"{video_folder}/crops"
            original_crops_folder = f"{crops_folder}/original"
            
            # Create a temporary directory for crop extraction
            temp_dir = f"/tmp/crops_{video_guid}"
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Load the saved detection results instead of re-running detection
                detections_blob = detection_result.get("detections_blob")
                if not detections_blob:
                    raise RuntimeError("No detections blob found in detection result")
                
                # Download the detection results
                temp_detections_path = f"/tmp/detections_{video_guid}.json"
                if not self.tenant_storage.download_blob(detections_blob, temp_detections_path):
                    raise RuntimeError(f"Failed to download detection results: {detections_blob}")
                
                # Load the detection results
                with open(temp_detections_path, 'r') as f:
                    saved_detections = json.load(f)
                
                logger.info(f"Loaded {len(saved_detections)} detection frames from storage")
                
                # Convert saved detections back to supervision.Detections format
                all_detections = []
                for frame_idx, frame_detection in enumerate(saved_detections):
                    if isinstance(frame_detection, dict) and "boxes" in frame_detection:
                        # Convert back to supervision.Detections format
                        boxes = np.array(frame_detection["boxes"])
                        confidence = np.array(frame_detection["confidence"])
                        class_id = np.array(frame_detection["class_id"])
                        
                        # Debug: Check the actual detection data
                        logger.info(f"Frame {frame_idx}: {len(boxes)} boxes, confidence range: {confidence.min():.3f}-{confidence.max():.3f}")
                        logger.info(f"  Sample box: {boxes[0] if len(boxes) > 0 else 'No boxes'}")
                        
                        # Create supervision.Detections object
                        detections = sv.Detections(
                            xyxy=boxes,
                            confidence=confidence,
                            class_id=class_id
                        )
                        
                        # Add frame_id to the detection data
                        if detections.data is None:
                            detections.data = {}
                        detections.data["frame_id"] = np.full(len(detections), frame_idx)
                        
                        all_detections.append(detections)
                        logger.debug(f"Converted frame {frame_idx}: {len(detections)} detections")
                    else:
                        logger.warning(f"Skipping frame {frame_idx}: Invalid detection format")
                
                if len(all_detections) == 0:
                    raise RuntimeError("No valid detections found in saved results")
                
                logger.info(f"Successfully converted {len(all_detections)} detection frames")
                
                # Debug: Log detection statistics
                total_detections = sum(len(det) for det in all_detections)
                logger.info(f"Total detections across all frames: {total_detections}")
                
                # Load the actual frame images from storage
                frames_folder = context.get("frames_folder")
                if not frames_folder:
                    # Use selected_frames folder instead
                    frames_folder = f"{video_folder}/selected_frames"
                
                # Download all frame images to a temporary directory
                temp_frames_dir = f"/tmp/frames_{video_guid}"
                os.makedirs(temp_frames_dir, exist_ok=True)
                
                # First, list all available frames in the selected_frames folder
                selected_frames_blobs = self.tenant_storage.list_blobs(frames_folder)
                available_frames = []
                for blob in selected_frames_blobs:
                    if blob.endswith('.jpg'):
                        # Extract frame number from filename like 'frame_375.jpg'
                        frame_name = blob.split('/')[-1]  # Get just the filename
                        if frame_name.startswith('frame_'):
                            frame_number = int(frame_name.split('_')[1].split('.')[0])
                            # Remove the user_path prefix from the blob path if present
                            user_path_prefix = f"{self.tenant_storage.config.user_path}/"
                            clean_blob_path = blob
                            if blob.startswith(user_path_prefix):
                                clean_blob_path = blob[len(user_path_prefix):]
                            available_frames.append((frame_number, clean_blob_path))
                
                # Sort by frame number
                available_frames.sort(key=lambda x: x[0])
                
                logger.info(f"Found {len(available_frames)} available frames for crop extraction")
                
                frame_images = []
                for frame_number, frame_blob in available_frames:
                    temp_frame_path = f"{temp_frames_dir}/frame_{frame_number}.jpg"
                    
                    if self.tenant_storage.download_blob(frame_blob, temp_frame_path):
                        # Load the frame image
                        frame_image = cv2.imread(temp_frame_path)
                        if frame_image is not None:
                            frame_images.append(frame_image)
                            logger.debug(f"Loaded frame {frame_number} from {frame_blob}")
                        else:
                            logger.warning(f"Failed to load frame image: {temp_frame_path}")
                    else:
                        logger.warning(f"Failed to download frame: {frame_blob}")
                
                if len(frame_images) == 0:
                    raise RuntimeError("No frame images could be loaded")
                
                logger.info(f"Loaded {len(frame_images)} frame images for crop extraction")
                
                # Create frame generator from the loaded frame images
                frame_generator = create_frame_generator_from_images(frame_images, input_format='BGR')
                
                # Extract crops using the crop extractor utility
                crops_dir, all_crops_dir = extract_crops_from_video(
                    frame_generator,
                    all_detections,
                    temp_dir,
                    crop_extract_interval=1  # Extract from all frames since we have limited frames
                )
                
                # OPTIMIZATION: Keep crops in memory for next step
                crops_in_memory = {}
                crops_uploaded = 0
                
                # Process the extracted crops and upload them to Google Storage
                crops_in_memory = {}
                crops_uploaded = 0
                
                if os.path.exists(all_crops_dir):
                    # Walk through all crop files and process them
                    for root, dirs, files in os.walk(all_crops_dir):
                        for file in files:
                            if file.endswith('.jpg'):
                                local_crop_path = os.path.join(root, file)
                                
                                # Calculate relative path from all_crops_dir
                                rel_path = os.path.relpath(local_crop_path, all_crops_dir)
                                
                                # Load crop into memory
                                crop_img = cv2.imread(local_crop_path)
                                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                                if crop_img is not None:
                                    crops_in_memory[rel_path] = crop_img
                                    
                                    # Upload to Google Storage
                                    storage_crop_path = f"{original_crops_folder}/{rel_path}"
                                    
                                    if self.tenant_storage.upload_from_file(storage_crop_path, local_crop_path):
                                        crops_uploaded += 1
                                        logger.debug(f"Uploaded crop: {storage_crop_path}")
                                    else:
                                        logger.warning(f"Failed to upload crop: {storage_crop_path}")
                else:
                    logger.warning(f"Local crops directory not found: {all_crops_dir}")
                
                logger.info(f"Successfully extracted and uploaded {crops_uploaded} crops for video: {video_guid}")
                logger.info(f"Loaded {len(crops_in_memory)} crops into memory for next step")
                
                return {
                    "crops_extracted": True,
                    "video_guid": video_guid,
                    "crops_folder": crops_folder,
                    "original_crops_folder": original_crops_folder,
                    "crops_uploaded": crops_uploaded,
                    "total_detections": len(all_detections),
                    "crops_in_memory": crops_in_memory  # OPTIMIZATION: Pass crops in memory to next step
                }
                
            finally:
                # Clean up temporary files
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temporary crops directory: {temp_dir}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary crops directory {temp_dir}: {cleanup_error}")
                
                # Clean up temporary detections file
                temp_detections_path = f"/tmp/detections_{video_guid}.json"
                if os.path.exists(temp_detections_path):
                    try:
                        os.remove(temp_detections_path)
                        logger.debug(f"Cleaned up temporary detections file: {temp_detections_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary detections file {temp_detections_path}: {cleanup_error}")
                        
        except Exception as e:
            logger.error(f"Failed to extract crops for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    
    def _remove_crop_background(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove background from player crops using the initialized grass mask detector.
        
        Args:
            context: Pipeline context containing crop extraction results
            
        Returns:
            Dictionary with background removal results
        """
        crops_extracted = context.get("crops_extracted", False)
        video_guid = context.get("video_guid")
        original_crops_folder = context.get("original_crops_folder")
        grass_mask_initialized = context.get("grass_mask_initialized", False)
        
        if not crops_extracted or not original_crops_folder:
            logger.warning("No crops extracted or original crops folder not found")
            return {"status": StepStatus.ERROR.value, "error": "No crops extracted or original crops folder not found"}
        
        if not grass_mask_initialized:
            logger.warning("Grass mask detector not initialized")
            return {"status": StepStatus.ERROR.value, "error": "Grass mask detector not initialized"}
        
        try:
            logger.info(f"Removing background from crops for video: {video_guid}")
            
            # Get the video folder and create modified crops folder
            video_folder = context.get("video_folder", f"{self.run_folder}/video_{video_guid}")
            crops_folder = f"{video_folder}/crops"
            modified_crops_folder = f"{crops_folder}/modified"
            
            # OPTIMIZATION: Use in-memory crops if available
            crops_in_memory = context.get("crops_in_memory", {})
            
            logger.info(f"Background removal - crops_in_memory available: {len(crops_in_memory)} crops")
            
            if crops_in_memory:
                logger.info(f"Using {len(crops_in_memory)} crops from memory (optimized)")
                
                # Log sample crop information for debugging
                if len(crops_in_memory) > 0:
                    sample_path, sample_crop = next(iter(crops_in_memory.items()))
                    logger.debug(f"Sample crop: {sample_path}, shape: {sample_crop.shape}, dtype: {sample_crop.dtype}")
                    logger.debug(f"Sample crop pixel [0,0]: {sample_crop[0,0]}")
                
                # Process crops directly from memory
                modified_crops_in_memory = {}
                crops_processed = 0
                
                # Create temporary directory for saving processed crops
                temp_dir = f"/tmp/background_removal_{video_guid}"
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    for rel_path, crop_img in crops_in_memory.items():
                        try:
                            # Remove background using the grass mask detector
                            # Note: crops_in_memory contains RGB images, so we specify RGB input format
                            processed_result = self.background_mask_detector.remove_background(
                                crop_img, 
                                input_format='RGB'
                            )
                            
                            # Ensure we have a single image (not a list)
                            if isinstance(processed_result, list):
                                processed_crop = processed_result[0]
                            else:
                                processed_crop = processed_result
                            
                            # Log background removal results for debugging
                            if crops_processed < 3:  # Only log first few crops to avoid spam
                                logger.debug(f"Crop {rel_path}: Original pixel [0,0]: {crop_img[0,0]}")
                                logger.debug(f"Crop {rel_path}: Processed pixel [0,0]: {processed_crop[0,0]}")
                            
                            # Store in memory for next step
                            modified_crops_in_memory[rel_path] = processed_crop
                            
                            # Parse crop information from rel_path (format: frame_{frame_id}/{frame_id}_{tracker_id}_{confidence}.jpg)
                            path_parts = rel_path.split('/')
                            if len(path_parts) >= 2:
                                frame_folder = path_parts[0]  # e.g., "frame_123"
                                crop_filename = path_parts[1]  # e.g., "123_456_0.850.jpg"
                                
                                # Extract frame_id and tracker_id from filename
                                name_parts = crop_filename.split('_')
                                if len(name_parts) >= 2:
                                    frame_id = name_parts[0]
                                    tracker_id = name_parts[1]
                                    
                                    # Create new path structure: video_folder/crops/modified/frame{frame_id}/crop_{tracker_id}/crop.jpg
                                    new_storage_path = f"{modified_crops_folder}/frame{frame_id}/crop_{tracker_id}/crop.jpg"
                                    
                                    # Save the processed crop to temporary location
                                    temp_modified_path = os.path.join(temp_dir, f"frame{frame_id}", f"crop_{tracker_id}", "crop.jpg")
                                    os.makedirs(os.path.dirname(temp_modified_path), exist_ok=True)
                                    
                                    # Convert RGB to BGR for OpenCV saving (which will result in RGB on disk)
                                    crop_bgr = cv2.cvtColor(processed_crop, cv2.COLOR_RGB2BGR)
                                    if cv2.imwrite(temp_modified_path, crop_bgr):
                                        # Upload the modified crop to Google Storage
                                        if self.tenant_storage.upload_from_file(new_storage_path, temp_modified_path):
                                            crops_processed += 1
                                            logger.debug(f"Processed and uploaded crop: {new_storage_path}")
                                        else:
                                            logger.warning(f"Failed to upload modified crop: {new_storage_path}")
                                    else:
                                        logger.warning(f"Failed to save modified crop: {temp_modified_path}")
                                else:
                                    logger.warning(f"Invalid crop filename format: {crop_filename}")
                            else:
                                logger.warning(f"Invalid crop path format: {rel_path}")
                                
                        except Exception as crop_error:
                            logger.warning(f"Failed to process crop {rel_path}: {crop_error}")
                            continue
                    
                    logger.info(f"Successfully processed {crops_processed} crops from memory for video: {video_guid}")
                    
                    return {
                        "background_removed": True,
                        "video_guid": video_guid,
                        "original_crops_folder": original_crops_folder,
                        "modified_crops_folder": modified_crops_folder,
                        "crops_processed": crops_processed,
                        "total_crops_processed": len(crops_in_memory),
                        "modified_crops_in_memory": modified_crops_in_memory,  # OPTIMIZATION: Pass to next step
                        "optimization_used": "in_memory_processing"
                    }
                    
                finally:
                    # Clean up temporary directory
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                            logger.debug(f"Cleaned up temporary background removal directory: {temp_dir}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {cleanup_error}")
            
            else:
                # FALLBACK: Original implementation using Google Storage
                logger.info("Using fallback implementation with Google Storage downloads")
                
                # Create temporary directory for processing
                temp_dir = f"/tmp/background_removal_{video_guid}"
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    # Download all original crops from Google Storage
                    temp_original_dir = os.path.join(temp_dir, "original")
                    os.makedirs(temp_original_dir, exist_ok=True)
                    
                    # List all blobs in the original crops folder
                    blobs = self.tenant_storage.list_blobs(prefix=f"{original_crops_folder}/")
                    
                    if not blobs:
                        logger.error(f"No blobs found in original crops folder: {original_crops_folder}")
                        return {"status": StepStatus.ERROR.value, "error": f"No crops found in {original_crops_folder}"}
                    
                    downloaded_crops = []
                    crops_processed = 0
                    failed_downloads = 0
                    
                    logger.info(f"Found {len(blobs)} blobs in original crops folder: {original_crops_folder}")
                    logger.debug(f"User path prefix: {self.tenant_storage.config.user_path}/")
                    logger.debug(f"Full crops folder path: {self.tenant_storage.config.user_path}/{original_crops_folder}")
                    
                    for blob_name in blobs:
                        if blob_name.endswith('.jpg'):
                            # Calculate local path maintaining directory structure
                            # blob_name includes user_path prefix, so we need to calculate rel_path correctly
                            user_path_prefix = f"{self.tenant_storage.config.user_path}/"
                            full_original_crops_folder = f"{user_path_prefix}{original_crops_folder}"
                            
                            if blob_name.startswith(f"{full_original_crops_folder}/"):
                                rel_path = blob_name[len(f"{full_original_crops_folder}/"):]
                            else:
                                # Fallback: try with just the original_crops_folder
                                rel_path = blob_name[len(f"{original_crops_folder}/"):]
                            
                            local_path = os.path.join(temp_original_dir, rel_path)
                            
                            # Create directory structure if needed
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            
                            # Debug: Log the blob path being downloaded
                            logger.debug(f"Attempting to download blob: {blob_name}")
                            
                            # IMPORTANT: The blob_name from list_blobs includes the user_path prefix,
                            # but download_blob expects a path without the user_path prefix
                            # So we need to strip the user_path prefix from the blob_name
                            user_path_prefix = f"{self.tenant_storage.config.user_path}/"
                            if blob_name.startswith(user_path_prefix):
                                blob_path_for_download = blob_name[len(user_path_prefix):]
                            else:
                                blob_path_for_download = blob_name
                            
                            logger.debug(f"Downloading blob path: {blob_path_for_download}")
                            
                            # Download the crop
                            if self.tenant_storage.download_blob(blob_path_for_download, local_path):
                                downloaded_crops.append((blob_name, local_path, rel_path))
                                logger.debug(f"Downloaded crop: {blob_name}")
                            else:
                                failed_downloads += 1
                                logger.warning(f"Failed to download crop: {blob_name}")
                                logger.warning(f"  Original blob name: {blob_name}")
                                logger.warning(f"  Download path used: {blob_path_for_download}")
                    
                    # Check if too many downloads failed - this indicates a systemic issue
                    if failed_downloads > 0 and len(downloaded_crops) == 0:
                        logger.error(f"Failed to download any crops from {original_crops_folder}. This indicates a path or storage issue.")
                        return {"status": StepStatus.ERROR.value, "error": f"Failed to download any crops from storage. Check paths and storage configuration."}
                    
                    if failed_downloads > len(downloaded_crops):
                        logger.error(f"More downloads failed ({failed_downloads}) than succeeded ({len(downloaded_crops)}). This indicates a systemic issue.")
                        return {"status": StepStatus.ERROR.value, "error": f"Majority of crop downloads failed. Check paths and storage configuration."}
                    
                    logger.info(f"Downloaded {len(downloaded_crops)} crops for background removal (failed: {failed_downloads})")
                    
                    # Process each crop to remove background
                    for blob_name, local_path, rel_path in downloaded_crops:
                        try:
                            # Read the crop image
                            crop_img = cv2.imread(local_path)
                            
                            if crop_img is None:
                                logger.warning(f"Failed to read crop image: {local_path}")
                                continue
                            
                            # Convert BGR to RGB since cv2.imread returns BGR but our crops are stored as RGB
                            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                            
                            # Remove background using the grass mask detector
                            processed_result = self.background_mask_detector.remove_background(
                                crop_rgb, 
                                input_format='RGB'
                            )
                            
                            # Ensure we have a single image (not a list)
                            if isinstance(processed_result, list):
                                processed_crop = processed_result[0]
                            else:
                                processed_crop = processed_result
                            
                            # Parse crop information from rel_path (format: frame_{frame_id}/{frame_id}_{tracker_id}_{confidence}.jpg)
                            path_parts = rel_path.split('/')
                            if len(path_parts) >= 2:
                                frame_folder = path_parts[0]  # e.g., "frame_123"
                                crop_filename = path_parts[1]  # e.g., "123_456_0.850.jpg"
                                
                                # Extract frame_id and tracker_id from filename
                                name_parts = crop_filename.split('_')
                                if len(name_parts) >= 2:
                                    frame_id = name_parts[0]
                                    tracker_id = name_parts[1]
                                    
                                    # Create new path structure: video_folder/crops/modified/frame{frame_id}/crop_{tracker_id}/crop.jpg
                                    new_storage_path = f"{modified_crops_folder}/frame{frame_id}/crop_{tracker_id}/crop.jpg"
                                    
                                    # Save the processed crop to temporary location
                                    temp_modified_path = os.path.join(temp_dir, "modified", f"frame{frame_id}", f"crop_{tracker_id}", "crop.jpg")
                                    os.makedirs(os.path.dirname(temp_modified_path), exist_ok=True)
                                    
                                    # Convert RGB to BGR for OpenCV saving (which will result in RGB on disk)
                                    crop_bgr = cv2.cvtColor(processed_crop, cv2.COLOR_RGB2BGR)
                                    if cv2.imwrite(temp_modified_path, crop_bgr):
                                        # Upload the modified crop to Google Storage
                                        if self.tenant_storage.upload_from_file(new_storage_path, temp_modified_path):
                                            crops_processed += 1
                                            logger.debug(f"Processed and uploaded crop: {new_storage_path}")
                                        else:
                                            logger.warning(f"Failed to upload modified crop: {new_storage_path}")
                                    else:
                                        logger.warning(f"Failed to save modified crop: {temp_modified_path}")
                                else:
                                    logger.warning(f"Invalid crop filename format: {crop_filename}")
                            else:
                                logger.warning(f"Invalid crop path format: {rel_path}")
                                
                        except Exception as crop_error:
                            logger.warning(f"Failed to process crop {local_path}: {crop_error}")
                            continue
                    
                    logger.info(f"Successfully removed background from {crops_processed} crops for video: {video_guid}")
                    
                    return {
                        "background_removed": True,
                        "video_guid": video_guid,
                        "original_crops_folder": original_crops_folder,
                        "modified_crops_folder": modified_crops_folder,
                        "crops_processed": crops_processed,
                        "total_crops_downloaded": len(downloaded_crops),
                        "optimization_used": "fallback_storage"
                    }
                    
                finally:
                    # Clean up temporary directory
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                            logger.debug(f"Cleaned up temporary background removal directory: {temp_dir}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {cleanup_error}")
                        
        except Exception as e:
            logger.error(f"Failed to remove background from crops for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    
    def _augment_crops(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augment player crops for training and store them in the same directories as source crops.
        
        Args:
            context: Pipeline context containing background removal results
            
        Returns:
            Dictionary with augmentation results
        """
        background_removed = context.get("background_removed", False)
        modified_crops_folder = context.get("modified_crops_folder")
        modified_crops_in_memory = context.get("modified_crops_in_memory", {})
        crops_processed = context.get("crops_processed", 0)
        video_guid = context.get("video_guid")
        
        if not background_removed or not modified_crops_folder or crops_processed == 0:
            logger.warning("No background-removed crops to augment")
            return {"crops_augmented": False, "error": "No background-removed crops available"}
        
        try:
            logger.info(f"Augmenting {crops_processed} crops for video: {video_guid}")
            logger.info(f"Using modified crops from: {modified_crops_folder}")
            
            # Create temporary directory for saving augmented crops
            temp_dir = f"/tmp/augmentation_{video_guid}"
            os.makedirs(temp_dir, exist_ok=True)
            
            augmented_crops_in_memory = {}
            total_augmented = 0
            source_crops = 0
            
            try:
                # Use in-memory crops if available, otherwise download from storage
                if modified_crops_in_memory:
                    logger.info(f"Using {len(modified_crops_in_memory)} crops from memory (optimized)")
                    crops_to_process = modified_crops_in_memory.items()
                    source_crops = len(modified_crops_in_memory)
                else:
                    logger.info("Downloading crops from storage for augmentation")
                    crops_to_process = self._download_crops_for_augmentation(modified_crops_folder, temp_dir)
                    source_crops = len(list(crops_to_process)) if hasattr(crops_to_process, '__len__') else 0
                
                # Process each crop for augmentation
                for rel_path, crop_img in crops_to_process:
                    try:
                        # Parse crop path to extract frame_id and tracker_id
                        frame_id, tracker_id = self._parse_crop_path(rel_path)
                        if not frame_id or not tracker_id:
                            logger.warning(f"Could not parse crop path: {rel_path}")
                            continue
                        
                        # Generate augmented images (crop_img is expected to be RGB format)
                        augmented_images = augment_images([crop_img])
                        
                        # Store original crop
                        augmented_crops_in_memory[rel_path] = crop_img
                        
                        # Process and store each augmented image
                        for i, aug_img in enumerate(augmented_images[1:], 1):  # Skip original at index 0
                            aug_storage_path = f"{modified_crops_folder}/frame{frame_id}/crop_{tracker_id}/crop_aug{i}.jpg"
                            aug_rel_path = f"frame{frame_id}/crop_{tracker_id}/crop_aug{i}.jpg"
                            
                            # Store in memory for next pipeline step
                            augmented_crops_in_memory[aug_rel_path] = aug_img
                            
                            # Save and upload augmented crop
                            if self._save_and_upload_crop(aug_img, aug_storage_path, temp_dir, frame_id, tracker_id, f"crop_aug{i}.jpg"):
                                total_augmented += 1
                            
                    except Exception as crop_error:
                        logger.warning(f"Failed to augment crop {rel_path}: {crop_error}")
                        continue
                
                logger.info(f"Successfully augmented {total_augmented} crops from {source_crops} source crops for video: {video_guid}")
                
                return {
                    "crops_augmented": True,
                    "video_guid": video_guid,
                    "modified_crops_folder": modified_crops_folder,
                    "total_augmented": total_augmented,
                    "source_crops": source_crops,
                    "augmented_crops_in_memory": augmented_crops_in_memory,
                    "optimization_used": "in_memory_processing" if modified_crops_in_memory else "fallback_storage"
                }
                
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temporary augmentation directory: {temp_dir}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {cleanup_error}")
            
        except Exception as e:
            logger.error(f"Failed to augment crops for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    
    def _download_crops_for_augmentation(self, modified_crops_folder: str, temp_dir: str):
        """
        Download crops from storage for augmentation processing.
        
        Args:
            modified_crops_folder: Storage path to modified crops
            temp_dir: Temporary directory for downloads
            
        Yields:
            Tuples of (rel_path, crop_image_rgb)
        """
        temp_crops_dir = os.path.join(temp_dir, "crops")
        os.makedirs(temp_crops_dir, exist_ok=True)
        
        # List all blobs in the modified crops folder
        blobs = self.tenant_storage.list_blobs(prefix=f"{modified_crops_folder}/")
        
        for blob_name in blobs:
            if blob_name.endswith('.jpg') and 'crop.jpg' in blob_name:  # Only process original crops
                # Calculate download path
                user_path_prefix = f"{self.tenant_storage.config.user_path}/"
                blob_path_for_download = blob_name[len(user_path_prefix):] if blob_name.startswith(user_path_prefix) else blob_name
                
                # Extract relative path
                if blob_name.startswith(f"{user_path_prefix}{modified_crops_folder}/"):
                    rel_path = blob_name[len(f"{user_path_prefix}{modified_crops_folder}/"):]
                else:
                    rel_path = blob_name[len(f"{modified_crops_folder}/"):]
                
                local_path = os.path.join(temp_crops_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Download and read the crop
                if self.tenant_storage.download_blob(blob_path_for_download, local_path):
                    crop_img = cv2.imread(local_path)
                    if crop_img is not None:
                        # Convert BGR to RGB (crops are stored as RGB but cv2.imread returns BGR)
                        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        yield rel_path, crop_rgb
                    else:
                        logger.warning(f"Failed to read crop image: {local_path}")
                else:
                    logger.warning(f"Failed to download crop: {blob_name}")
    
    def _parse_crop_path(self, rel_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse crop path to extract frame_id and tracker_id.
        
        Args:
            rel_path: Relative path like "frame_123/123_456_0.850.jpg" or "frame123/crop_456/crop.jpg"
            
        Returns:
            Tuple of (frame_id, tracker_id) or (None, None) if parsing fails
        """
        path_parts = rel_path.split('/')
        
        # Handle old format: frame_123/123_456_0.850.jpg
        if len(path_parts) >= 2 and path_parts[0].startswith('frame_'):
            frame_folder = path_parts[0]  # e.g., "frame_123"
            crop_filename = path_parts[1]  # e.g., "123_456_0.850.jpg"
            
            # Extract frame_id and tracker_id from filename
            name_parts = crop_filename.split('_')
            if len(name_parts) >= 2:
                return name_parts[0], name_parts[1]
        
        # Handle new format: frame123/crop_456/crop.jpg
        elif len(path_parts) >= 3:
            frame_folder = path_parts[0]  # e.g., "frame123"
            crop_folder = path_parts[1]   # e.g., "crop_456"
            
            frame_id = frame_folder.replace('frame', '')
            tracker_id = crop_folder.replace('crop_', '')
            return frame_id, tracker_id
        
        return None, None
    
    def _save_and_upload_crop(self, crop_img: np.ndarray, storage_path: str, temp_dir: str, 
                             frame_id: str, tracker_id: str, filename: str) -> bool:
        """
        Save crop image to temporary location and upload to storage.
        
        Args:
            crop_img: RGB crop image
            storage_path: Full storage path for upload
            temp_dir: Temporary directory
            frame_id: Frame identifier
            tracker_id: Tracker identifier  
            filename: Final filename
            
        Returns:
            True if successful, False otherwise
        """
        temp_path = os.path.join(temp_dir, f"frame{frame_id}", f"crop_{tracker_id}", filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Convert RGB to BGR for OpenCV saving (results in correct RGB on disk)
        crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
        
        if cv2.imwrite(temp_path, crop_bgr):
            if self.tenant_storage.upload_from_file(storage_path, temp_path):
                logger.debug(f"Saved and uploaded crop: {storage_path}")
                return True
            else:
                logger.warning(f"Failed to upload crop: {storage_path}")
        else:
            logger.warning(f"Failed to save crop: {temp_path}")
        
        return False
    
    def _create_training_and_validation_sets(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create training and validation datasets from processed crops.
        
        Args:
            context: Pipeline context containing augmentation results
            
        Returns:
            Dictionary with dataset creation results
        """
        crops_augmented = context.get("crops_augmented", False)
        modified_crops_folder = context.get("modified_crops_folder")
        video_guid = context.get("video_guid")
        
        if not crops_augmented or not modified_crops_folder:
            logger.warning("No augmented crops or modified crops folder to create datasets from")
            # Still need to return datasets_folder for consistency
            video_folder = context.get("video_folder", f"{self.run_folder}/video_{video_guid}")
            datasets_folder = f"{video_folder}/datasets"
            return {"datasets_created": False, "error": "No augmented crops available", "datasets_folder": datasets_folder}
        
        try:
            logger.info(f"Creating training and validation sets for video: {video_guid}")
            
            # Get the video folder and create datasets folder
            video_folder = context.get("video_folder", f"{self.run_folder}/video_{video_guid}")
            datasets_folder = f"{video_folder}/datasets"
            
            # OPTIMIZATION: Use in-memory crops if available (now includes augmented crops)
            augmented_crops_in_memory = context.get("augmented_crops_in_memory", {})
            
            if augmented_crops_in_memory:
                logger.info(f"Using {len(augmented_crops_in_memory)} augmented crops from memory (optimized)")
                
                # Create temporary directory for processing
                temp_dir = f"/tmp/datasets_{video_guid}"
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    # Organize crops by frame for frame-specific train/val splits
                    crops_by_frame = {}
                    track_folders = set()
                    
                    # Group crops by frame
                    for rel_path, crop_img in augmented_crops_in_memory.items():
                        # Extract frame information from path 
                        # Format: frame123/crop_456/crop.jpg or frame123/crop_456/crop_aug1.jpg
                        path_parts = rel_path.split('/')
                        if len(path_parts) >= 2:
                            frame_id = path_parts[0]  # e.g., "frame123"
                            crop_folder = path_parts[1]  # e.g., "crop_456"
                            track_folder = f"{frame_id}_{crop_folder}"
                        else:
                            frame_id = 'unknown_frame'
                            track_folder = 'unknown'
                        
                        track_folders.add(track_folder)
                        
                        if frame_id not in crops_by_frame:
                            crops_by_frame[frame_id] = {}
                        crops_by_frame[frame_id][rel_path] = crop_img
                    
                    logger.info(f"Prepared {len(augmented_crops_in_memory)} crops from {len(track_folders)} tracks across {len(crops_by_frame)} frames")
                    
                    # Process each frame separately
                    total_train_samples = 0
                    total_val_samples = 0
                    
                    for frame_id, frame_crops in crops_by_frame.items():
                        # Skip frames with no crops
                        if not frame_crops:
                            logger.warning(f"Skipping frame {frame_id} - no crops available")
                            continue
                            
                        logger.info(f"Processing frame {frame_id} with {len(frame_crops)} crops")
                        
                        # Create temporary frame directory
                        temp_frame_dir = os.path.join(temp_dir, frame_id)
                        temp_crops_dir = os.path.join(temp_frame_dir, "crops")
                        os.makedirs(temp_crops_dir, exist_ok=True)
                        
                        # Save crops for this frame to temporary directory
                        saved_crops_count = 0
                        for rel_path, crop_img in frame_crops.items():
                            # Use just the crop part of the path (remove frame prefix)
                            crop_local_path = '/'.join(rel_path.split('/')[1:])  # Remove "frame123/" part
                            local_path = os.path.join(temp_crops_dir, crop_local_path)
                            
                            # Create directory structure if needed
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            
                            # Save crop to temporary location
                            # Convert RGB to BGR for OpenCV saving (which will result in RGB on disk)
                            crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                            if cv2.imwrite(local_path, crop_bgr):
                                saved_crops_count += 1
                                logger.debug(f"Saved crop: {local_path}")
                            else:
                                logger.warning(f"Failed to save crop: {local_path}")
                        
                        logger.info(f"Saved {saved_crops_count} crops to {temp_crops_dir}")
                        
                        # Debug: List the actual structure created
                        if os.path.exists(temp_crops_dir):
                            items_in_crops_dir = os.listdir(temp_crops_dir)
                            logger.info(f"Items in {temp_crops_dir}: {items_in_crops_dir}")
                        else:
                            logger.warning(f"Crops directory doesn't exist: {temp_crops_dir}")
                        
                        # Create train/val split for this frame
                        temp_frame_datasets_dir = os.path.join(temp_frame_dir, "datasets")
                        
                        # Only attempt train/val split if we have crops
                        if saved_crops_count > 0:
                            try:
                                create_train_val_split(
                                    source_folder=temp_crops_dir,
                                    destin_folder=temp_frame_datasets_dir,
                                    train_ratio=0.8
                                )
                                logger.info(f"Successfully created train/val split for frame {frame_id}")
                            except Exception as split_error:
                                logger.warning(f"Failed to create train/val split for frame {frame_id}: {split_error}")
                                continue
                        else:
                            logger.warning(f"No crops saved for frame {frame_id}, skipping train/val split")
                        
                        # Upload frame-specific train and val datasets to Google Storage
                        frame_train_samples = 0
                        frame_val_samples = 0
                        
                        # Upload train dataset for this frame
                        train_local_dir = os.path.join(temp_frame_datasets_dir, "train")
                        if os.path.exists(train_local_dir):
                            for root, dirs, files in os.walk(train_local_dir):
                                for file in files:
                                    if file.endswith('.jpg'):
                                        local_path = os.path.join(root, file)
                                        rel_path = os.path.relpath(local_path, train_local_dir)
                                        storage_path = f"{datasets_folder}/{frame_id}/train/{rel_path}"
                                        
                                        if self.tenant_storage.upload_from_file(storage_path, local_path):
                                            frame_train_samples += 1
                                            logger.debug(f"Uploaded train sample: {storage_path}")
                                        else:
                                            logger.warning(f"Failed to upload train sample: {storage_path}")
                        
                        # Upload val dataset for this frame
                        val_local_dir = os.path.join(temp_frame_datasets_dir, "val")
                        if os.path.exists(val_local_dir):
                            for root, dirs, files in os.walk(val_local_dir):
                                for file in files:
                                    if file.endswith('.jpg'):
                                        local_path = os.path.join(root, file)
                                        rel_path = os.path.relpath(local_path, val_local_dir)
                                        storage_path = f"{datasets_folder}/{frame_id}/val/{rel_path}"
                                        
                                        if self.tenant_storage.upload_from_file(storage_path, local_path):
                                            frame_val_samples += 1
                                            logger.debug(f"Uploaded val sample: {storage_path}")
                                        else:
                                            logger.warning(f"Failed to upload val sample: {storage_path}")
                        
                        total_train_samples += frame_train_samples
                        total_val_samples += frame_val_samples
                        
                        logger.info(f"Frame {frame_id}: {frame_train_samples} train samples, {frame_val_samples} val samples")
                    
                    logger.info(f"Successfully created training and validation sets for video: {video_guid}")
                    logger.info(f"Total training samples: {total_train_samples}, Total validation samples: {total_val_samples}")
                    
                    return {
                        "datasets_created": True,
                        "video_guid": video_guid,
                        "datasets_folder": datasets_folder,
                        "training_samples": total_train_samples,
                        "validation_samples": total_val_samples,
                        "frames_processed": len(crops_by_frame),
                        "total_crops_processed": len(augmented_crops_in_memory),
                        "tracks_processed": len(track_folders),
                        "optimization_used": "in_memory_processing"
                    }
                    
                finally:
                    # Clean up temporary directory
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                            logger.debug(f"Cleaned up temporary datasets directory: {temp_dir}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {cleanup_error}")
            
            else:
                # FALLBACK: Original implementation using Google Storage
                logger.info("Using fallback implementation with Google Storage downloads")
                
                # Create temporary directory for processing
                temp_dir = f"/tmp/datasets_{video_guid}"
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    # Download all modified crops from Google Storage
                    temp_crops_dir = os.path.join(temp_dir, "crops")
                    os.makedirs(temp_crops_dir, exist_ok=True)
                    
                    # List all blobs in the modified crops folder (now includes augmented crops)
                    blobs = self.tenant_storage.list_blobs(prefix=f"{modified_crops_folder}/")
                    
                    # Organize crops by frame
                    crops_by_frame = {}
                    track_folders = set()
                    
                    for blob_name in blobs:
                        if blob_name.endswith('.jpg'):
                            # Calculate relative path
                            user_path_prefix = f"{self.tenant_storage.config.user_path}/"
                            if blob_name.startswith(user_path_prefix):
                                blob_path_for_download = blob_name[len(user_path_prefix):]
                            else:
                                blob_path_for_download = blob_name
                            
                            # Extract the relative path structure
                            if blob_name.startswith(f"{user_path_prefix}{modified_crops_folder}/"):
                                rel_path = blob_name[len(f"{user_path_prefix}{modified_crops_folder}/"):]
                            else:
                                rel_path = blob_name[len(f"{modified_crops_folder}/"):]
                            
                            # Extract frame information from path
                            path_parts = rel_path.split('/')
                            if len(path_parts) >= 2:
                                frame_id = path_parts[0]  # e.g., "frame123"
                                crop_folder = path_parts[1]  # e.g., "crop_456"
                                track_folder = f"{frame_id}_{crop_folder}"
                            else:
                                frame_id = 'unknown_frame'
                                track_folder = 'unknown'
                            
                            track_folders.add(track_folder)
                            
                            if frame_id not in crops_by_frame:
                                crops_by_frame[frame_id] = []
                            
                            crops_by_frame[frame_id].append((blob_name, blob_path_for_download, rel_path))
                    
                    logger.info(f"Found {sum(len(crops) for crops in crops_by_frame.values())} augmented crops across {len(crops_by_frame)} frames from {len(track_folders)} tracks")
                    
                    # Process each frame separately
                    total_train_samples = 0
                    total_val_samples = 0
                    
                    for frame_id, frame_crop_blobs in crops_by_frame.items():
                        # Skip frames with no crops
                        if not frame_crop_blobs:
                            logger.warning(f"Skipping frame {frame_id} - no crops available")
                            continue
                            
                        logger.info(f"Processing frame {frame_id} with {len(frame_crop_blobs)} crops")
                        
                        # Create temporary frame directory
                        temp_frame_dir = os.path.join(temp_dir, frame_id)
                        temp_frame_crops_dir = os.path.join(temp_frame_dir, "crops")
                        os.makedirs(temp_frame_crops_dir, exist_ok=True)
                        
                        # Download crops for this frame
                        downloaded_crops = []
                        for blob_name, blob_path_for_download, rel_path in frame_crop_blobs:
                            # Use just the crop part of the path (remove frame prefix)
                            crop_local_path = '/'.join(rel_path.split('/')[1:])  # Remove "frame123/" part
                            local_path = os.path.join(temp_frame_crops_dir, crop_local_path)
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            
                            # Download the crop
                            if self.tenant_storage.download_blob(blob_path_for_download, local_path):
                                downloaded_crops.append((blob_name, local_path, crop_local_path))
                                logger.debug(f"Downloaded crop: {blob_name}")
                            else:
                                logger.warning(f"Failed to download crop: {blob_name}")
                                logger.warning(f"  Original blob name: {blob_name}")
                                logger.warning(f"  Download path used: {blob_path_for_download}")
                        
                        logger.info(f"Downloaded {len(downloaded_crops)} crops for frame {frame_id}")
                        
                        # Create train/val split for this frame
                        temp_frame_datasets_dir = os.path.join(temp_frame_dir, "datasets")
                        
                        # Only attempt train/val split if we have crops
                        if len(downloaded_crops) > 0:
                            try:
                                create_train_val_split(
                                    source_folder=temp_frame_crops_dir,
                                    destin_folder=temp_frame_datasets_dir,
                                    train_ratio=0.8
                                )
                                logger.info(f"Successfully created train/val split for frame {frame_id}")
                            except Exception as split_error:
                                logger.warning(f"Failed to create train/val split for frame {frame_id}: {split_error}")
                                continue
                        else:
                            logger.warning(f"No crops downloaded for frame {frame_id}, skipping train/val split")
                        
                        # Upload frame-specific train and val datasets to Google Storage
                        frame_train_samples = 0
                        frame_val_samples = 0
                        
                        # Upload train dataset for this frame
                        train_local_dir = os.path.join(temp_frame_datasets_dir, "train")
                        if os.path.exists(train_local_dir):
                            for root, dirs, files in os.walk(train_local_dir):
                                for file in files:
                                    if file.endswith('.jpg'):
                                        local_path = os.path.join(root, file)
                                        rel_path = os.path.relpath(local_path, train_local_dir)
                                        storage_path = f"{datasets_folder}/{frame_id}/train/{rel_path}"
                                        
                                        if self.tenant_storage.upload_from_file(storage_path, local_path):
                                            frame_train_samples += 1
                                            logger.debug(f"Uploaded train sample: {storage_path}")
                                        else:
                                            logger.warning(f"Failed to upload train sample: {storage_path}")
                        
                        # Upload val dataset for this frame
                        val_local_dir = os.path.join(temp_frame_datasets_dir, "val")
                        if os.path.exists(val_local_dir):
                            for root, dirs, files in os.walk(val_local_dir):
                                for file in files:
                                    if file.endswith('.jpg'):
                                        local_path = os.path.join(root, file)
                                        rel_path = os.path.relpath(local_path, val_local_dir)
                                        storage_path = f"{datasets_folder}/{frame_id}/val/{rel_path}"
                                        
                                        if self.tenant_storage.upload_from_file(storage_path, local_path):
                                            frame_val_samples += 1
                                            logger.debug(f"Uploaded val sample: {storage_path}")
                                        else:
                                            logger.warning(f"Failed to upload val sample: {storage_path}")
                        
                        total_train_samples += frame_train_samples
                        total_val_samples += frame_val_samples
                        
                        logger.info(f"Frame {frame_id}: {frame_train_samples} train samples, {frame_val_samples} val samples")
                    
                    logger.info(f"Successfully created training and validation sets for video: {video_guid}")
                    logger.info(f"Total training samples: {total_train_samples}, Total validation samples: {total_val_samples}")
                    
                    return {
                        "datasets_created": True,
                        "video_guid": video_guid,
                        "datasets_folder": datasets_folder,
                        "training_samples": total_train_samples,
                        "validation_samples": total_val_samples,
                        "frames_processed": len(crops_by_frame),
                        "total_crops_processed": sum(len(crops) for crops in crops_by_frame.values()),
                        "tracks_processed": len(track_folders),
                        "optimization_used": "fallback_storage"
                    }
                    
                finally:
                    # Clean up temporary directory
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                            logger.debug(f"Cleaned up temporary datasets directory: {temp_dir}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {cleanup_error}")
                        
        except Exception as e:
            logger.error(f"Failed to create training and validation sets for video {video_guid}: {e}")
            # Still need to return datasets_folder for consistency
            video_folder = context.get("video_folder", f"{self.run_folder}/video_{video_guid}")
            datasets_folder = f"{video_folder}/datasets"
            return {"datasets_created": False, "error": str(e), "datasets_folder": datasets_folder}
    
    
    def _validate_video_resolution(self, video_path: str) -> bool:
        """
        Validate that video resolution is at least 1920 by 1080 pixels.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if resolution is adequate, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                return False
            
            # Check if file is not empty
            if os.path.getsize(video_path) == 0:
                logger.error(f"Video file is empty: {video_path}")
                return False
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"OpenCV could not open video file: {video_path}")
                logger.error("This could be due to:")
                logger.error("1. Unsupported video format")
                logger.error("2. Corrupted video file")
                logger.error("3. Missing codecs")
                logger.error("4. File permissions issue")
                return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            is_valid = width >= 1920 and height >= 1080
            logger.info(f"Video resolution: {width}x{height}, Valid: {is_valid}")
            
            if not is_valid:
                logger.error(f"Video resolution {width}x{height} is below minimum requirement of 1920x1080")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating video resolution for {video_path}: {e}")
            return False
    
    def _extract_video_metadata(self, video_path: str, original_blob_name: str, video_guid: str) -> Dict[str, Any]:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to the video file
            original_blob_name: Original blob name in Google Storage
            video_guid: Generated GUID for the video
            
        Returns:
            Dictionary with video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for metadata extraction: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        metadata = {
            "video_guid": video_guid,
            "original_name": os.path.basename(original_blob_name),
            "original_blob_name": original_blob_name,
            "processed_name": f"video_{video_guid}.mp4",
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "processed_at": datetime.now().isoformat()
        }
        
        return metadata

    def _extract_frames_for_detections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract frames on which to run detections.

        Args:
            context: Pipeline context containing loaded_video
            
        Returns:
            Dictionary with extracted frames data
        """
        loaded_video = context.get("loaded_video")
        
        if not loaded_video:
            logger.error("No loaded video for frame extraction - previous step failed")
            return {"status": StepStatus.ERROR.value, "error": "No loaded video provided - video loading failed"}
        
        temp_path = loaded_video["temp_path"]
        video_guid = loaded_video["video_guid"]
        
        try:
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for frame extraction: {temp_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                raise RuntimeError(f"Video has no frames or invalid frame count: {total_frames}")
            
            # Calculate initial frame positions (frames_per_video frames equally spaced)
            frame_positions = [int(i * total_frames / self.frames_per_video) for i in range(self.frames_per_video)]
            
            frames_data = []
            frame_ids = []
            
            # Create selected_frames directory path
            video_folder = context.get("video_folder", f"{self.run_folder}/video_{video_guid}")
            selected_frames_folder = f"{video_folder}/selected_frames"
            
            for frame_id in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if ret:
                    frames_data.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_ids.append(frame_id)
                    
                    # Save frame to Google Storage
                    frame_filename = f"frame_{frame_id}.jpg"
                    frame_blob_path = f"{selected_frames_folder}/{frame_filename}"
                    
                    # Create temporary file for the frame
                    temp_frame_path = f"/tmp/frame_{video_guid}_{frame_id}.jpg"
                    try:
                        # Save frame as JPEG
                        cv2.imwrite(temp_frame_path, frame)
                        
                        # Upload to Google Storage
                        if not self.tenant_storage.upload_from_file(frame_blob_path, temp_frame_path):
                            logger.warning(f"Failed to upload frame {frame_id} to storage")
                        else:
                            logger.debug(f"Saved frame {frame_id} to {frame_blob_path}")
                    except Exception as frame_error:
                        logger.warning(f"Failed to save frame {frame_id}: {frame_error}")
                    finally:
                        # Clean up temporary frame file
                        if os.path.exists(temp_frame_path):
                            try:
                                os.remove(temp_frame_path)
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to cleanup temp frame file {temp_frame_path}: {cleanup_error}")
            
            cap.release()

            if len(frames_data) == 0:
                raise RuntimeError(f"No frames could be extracted from video")

            logger.info(f"Extracted and saved {len(frames_data)} frames to {selected_frames_folder}")
            
            return {
                "frames_data": frames_data,
                "video_guid": video_guid,
                "frames_extracted": frame_ids,
                "selected_frames_folder": selected_frames_folder,
                "frames_saved_count": len(frames_data)
            }
            
        except Exception as e:
            logger.error(f"Critical error extracting frames from video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
        finally:
            # Clean up temporary file after frame extraction
            loaded_video = context.get("loaded_video", {})
            temp_path = loaded_video.get("temp_path")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file {temp_path}: {cleanup_error}")


    def _initialize_grass_mask(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the grass mask detector using the extracted frames.
        
        Args:
            context: Pipeline context containing frames_data
            
        Returns:
            Dictionary with grass mask initialization results
        """
        frames_data = context.get("frames_data")
        video_guid = context.get("video_guid")
        
        if not frames_data:
            # If frames_data is missing (e.g., when resuming from checkpoint), try to regenerate it
            logger.warning("No frames data for grass mask initialization - attempting to regenerate from video")
            
            # Try to get loaded_video info to re-extract frames
            loaded_video = context.get("loaded_video")
            if loaded_video and loaded_video.get("temp_path"):
                logger.info("Attempting to re-extract frames for grass mask initialization")
                # Re-extract frames for this step
                frames_result = self._extract_frames_for_detections(context)
                if frames_result.get("status") == StepStatus.ERROR.value:
                    return frames_result
                frames_data = frames_result.get("frames_data")
                # Update context with re-extracted frames
                context["frames_data"] = frames_data
            
            if not frames_data:
                logger.warning("No frames data for grass mask initialization")
                return {"status": StepStatus.ERROR.value, "error": "No frames data provided"}
        
        try:
            logger.info(f"Initializing grass mask detector for video: {video_guid}")
            
            # Create frame generator from the extracted frames
            frame_generator = create_frame_generator_from_images(frames_data, input_format='RGB')
            
            # Initialize the background mask detector with the frames
            self.background_mask_detector.initialize(frame_generator)
            
            # Get background statistics for logging
            stats = self.background_mask_detector.get_stats()
            
            logger.info(f"Grass mask detector initialized successfully for video: {video_guid}")
            logger.debug(f"Background color statistics: {stats}")
            
            # Convert numpy arrays to lists for JSON serialization
            background_stats = {}
            for key, value in stats.items():
                if hasattr(value, 'tolist'):  # numpy array
                    background_stats[key] = value.tolist()
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):  # other iterables
                    try:
                        background_stats[key] = list(value)
                    except:
                        background_stats[key] = str(value)
                else:
                    background_stats[key] = value
            
            return {
                "grass_mask_initialized": True,
                "video_guid": video_guid,
                "background_stats": background_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize grass mask detector for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}


def run_dataprep_pipeline(tenant_id: str = "tenant1", video_path: str = "", delete_original_raw_videos: bool = False, frames_per_video: int = 20, verbose: bool = False, save_intermediate: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run the data preparation pipeline.

    Args:
        tenant_id: The tenant ID to process videos for (default: "tenant1")
        video_path: Path to the video file to process
        delete_original_raw_videos: Whether to delete original raw video files after processing (default: False)
        frames_per_video: Number of frames to extract per video (default: 20)
        verbose: Whether to enable verbose logging (default: False)
        save_intermediate: Whether to save intermediate results (default: False)
        
    Returns:
        Dictionary with pipeline results
    """
    config = DetectionConfig()
    config.delete_original_raw_videos = delete_original_raw_videos
    config.frames_per_video = frames_per_video
    pipeline = DataPrepPipeline(config, tenant_id=tenant_id, verbose=verbose, save_intermediate=save_intermediate)

    # Ensure that the video path is correct
    if not video_path:
        raise ValueError("Video path must be provided")

    return pipeline.run(video_path)



if __name__ == "__main__":

    # Example usage
    tenant_id = "tenant1"
    delete_original_raw_videos = False  # Set to True to delete original raw videos after processing
    frames_per_video = 20  # Number of frames to extract per video
    verbose = True  # Enable verbose logging
    save_intermediate = True  # Save intermediate results
    
    # Get the first video from the raw directory
    try:
        # Get storage client for the tenant
        tenant_storage = get_storage(f"{tenant_id}/user")
        
        # List all blobs in the raw directory
        raw_blobs = tenant_storage.list_blobs(prefix="raw/")
        
        # Find the first MP4 video file
        video_path = None
        for blob_name in raw_blobs:
            if blob_name.endswith('.mp4') and blob_name != "raw/":
                # Extract just the filename from the full path
                video_path = blob_name.split("/")[-1]
                break
        
        if not video_path:
            print("No MP4 videos found in the raw directory")
            exit(1)
        
        print(f"Processing first video found: {video_path}")
        
    except Exception as e:
        print(f"Error accessing raw directory: {e}")
        # Fallback to a default video name
        video_path = "GRIT Dallas-Houston 2027 vs Team 91 2027 National - 7-30am_summary.mp4"
        print(f"Using fallback video: {video_path}")

    results = run_dataprep_pipeline(tenant_id=tenant_id, video_path=video_path, delete_original_raw_videos=delete_original_raw_videos, frames_per_video=frames_per_video, verbose=verbose, save_intermediate=save_intermediate)
    
    print("Pipeline Results:")
    print(json.dumps(results, indent=2))
