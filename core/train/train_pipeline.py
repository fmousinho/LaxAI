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
from config.all_config import DetectionConfig
from config import logging_config
from core.common.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
from core.common.crop_extractor import extract_crops_from_video
from core.train.augmentation import augment_images

logger = logging.getLogger(__name__)


class TrainPipeline(Pipeline):
    """
    Training pipeline that processes MP4 videos from Google Storage.
    
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
    
    def run(self, video_path: str) -> Dict[str, Any]:
        """
        Execute the complete training pipeline for a single video.
        
        Args:
            video_path: Path to the video file to process (can be local path or gs:// URL)
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        if not video_path:
            return {"status": PipelineStatus.ERROR.value, "error": "No video path provided"}

        logger.info(f"Processing video: {video_path}")

        # Create initial context with the video path
        initial_context = {"video_path": video_path}
        
        # Call the base class run method with the initial context
        results = super().run(initial_context)
        
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

            # Check if it's in the cloud or local
            if video_path.startswith("gs://"):
                imported_video = self._import_single_video(video_path, video_guid, video_folder)
            else:
                imported_video = video_path

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
            logger.warning("No imported video to load")
            return {"status": StepStatus.ERROR.value, "error": "No imported video provided"}
        
        # Create a robust temporary file path
        temp_video_path = f"/tmp/processing_{video_guid}.mp4"
        
        try:
            # If it's a Google Storage path, download it
            if imported_video.startswith("gs://") or "/" in imported_video:
                if not self.tenant_storage.download_blob(imported_video, temp_video_path):
                    raise RuntimeError(f"Failed to download video for processing: {imported_video}")
            else:
                # It's a local file, use it directly
                temp_video_path = imported_video
            
            # Validate video resolution
            if not self._validate_video_resolution(temp_video_path):
                raise RuntimeError(f"Video resolution too low: {imported_video}")
            
            logger.info(f"Loaded video for processing: {imported_video}")
            
            return {
                "loaded_video": {
                    "video_path": imported_video,
                    "video_guid": video_guid,
                    "temp_path": temp_video_path
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load video {imported_video}: {e}")
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
            logger.warning("No frames data for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No frames data provided"}
        
        try:
            detections = self.detection_model.generate_detections(frames_data)

            if len(detections) == 0:
                raise RuntimeError(f"Could not extract {self.frames_per_video} frames with sufficient detections")

            # Save detections
            video_folder = context.get("video_folder", f"{self.run_folder}/video_{video_guid}")
            detections_blob = f"{video_folder}/detections.json"

            detections_content = json.dumps(detections, indent=2)
            if not self.tenant_storage.upload_from_string(detections_blob, detections_content):
                raise RuntimeError(f"Failed to save detections for video: {video_guid}")
            
            logger.info(f"Completed detection for video: {video_guid}")
            
            return {
                "detection_result": {
                    "video_guid": video_guid,
                    "frames_data": frames_data,
                    "detections_blob": detections_blob
                }
            }
            
        except Exception as e:
            logger.error(f"Failed player detection for video {video_guid}: {e}")
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
                # Create frame generator from the extracted frames
                frame_generator = create_frame_generator_from_images(frames_data, input_format='BGR')
                
                # We need to run detection on each frame individually to get per-frame detections
                # Since the detection model doesn't currently support batch processing,
                # we'll process each frame individually
                all_detections = []
                frame_idx = 0
                
                for frame in frames_data:
                    # Run detection on this single frame
                    frame_detections = self.detection_model.generate_detections(frame)
                    
                    # Add frame_id to the detection data
                    if frame_detections.data is None:
                        frame_detections.data = {}
                    frame_detections.data["frame_id"] = np.full(len(frame_detections), frame_idx)
                    
                    all_detections.append(frame_detections)
                    frame_idx += 1
                
                # Create frame generator again (since the first one was consumed)
                frame_generator = create_frame_generator_from_images(frames_data, input_format='BGR')
                
                # Extract crops using the crop extractor utility
                crops_dir, all_crops_dir = extract_crops_from_video(
                    frame_generator,
                    all_detections,
                    temp_dir,
                    crop_extract_interval=1  # Extract from all frames since we have limited frames
                )
                
                # Upload crops to Google Storage
                crops_uploaded = 0
                if os.path.exists(all_crops_dir):
                    # Walk through all crop files and upload them
                    for root, dirs, files in os.walk(all_crops_dir):
                        for file in files:
                            if file.endswith('.jpg'):
                                local_crop_path = os.path.join(root, file)
                                
                                # Calculate relative path from all_crops_dir
                                rel_path = os.path.relpath(local_crop_path, all_crops_dir)
                                
                                # Upload to Google Storage
                                storage_crop_path = f"{original_crops_folder}/{rel_path}"
                                
                                if self.tenant_storage.upload_from_file(storage_crop_path, local_crop_path):
                                    crops_uploaded += 1
                                    logger.debug(f"Uploaded crop: {storage_crop_path}")
                                else:
                                    logger.warning(f"Failed to upload crop: {storage_crop_path}")
                
                logger.info(f"Successfully extracted and uploaded {crops_uploaded} crops for video: {video_guid}")
                
                return {
                    "crops_extracted": True,
                    "video_guid": video_guid,
                    "crops_folder": crops_folder,
                    "original_crops_folder": original_crops_folder,
                    "crops_uploaded": crops_uploaded,
                    "total_detections": len(all_detections)
                }
                
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temporary crops directory: {temp_dir}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary crops directory {temp_dir}: {cleanup_error}")
                        
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
            
            # Create temporary directory for processing
            temp_dir = f"/tmp/background_removal_{video_guid}"
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Download all original crops from Google Storage
                temp_original_dir = os.path.join(temp_dir, "original")
                os.makedirs(temp_original_dir, exist_ok=True)
                
                # List all blobs in the original crops folder
                blobs = self.tenant_storage.list_blobs(prefix=f"{original_crops_folder}/")
                
                downloaded_crops = []
                crops_processed = 0
                
                for blob_name in blobs:
                    if blob_name.endswith('.jpg'):
                        # Calculate local path maintaining directory structure
                        rel_path = blob_name[len(f"{original_crops_folder}/"):]
                        local_path = os.path.join(temp_original_dir, rel_path)
                        
                        # Create directory structure if needed
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        
                        # Download the crop
                        if self.tenant_storage.download_blob(blob_name, local_path):
                            downloaded_crops.append((blob_name, local_path, rel_path))
                            logger.debug(f"Downloaded crop: {blob_name}")
                        else:
                            logger.warning(f"Failed to download crop: {blob_name}")
                
                logger.info(f"Downloaded {len(downloaded_crops)} crops for background removal")
                
                # Process each crop to remove background
                for blob_name, local_path, rel_path in downloaded_crops:
                    try:
                        # Read the crop image
                        crop_img = cv2.imread(local_path)
                        
                        if crop_img is None:
                            logger.warning(f"Failed to read crop image: {local_path}")
                            continue
                        
                        # Remove background using the grass mask detector
                        processed_result = self.background_mask_detector.remove_background(
                            crop_img, 
                            input_format='BGR'
                        )
                        
                        # Ensure we have a single image (not a list)
                        if isinstance(processed_result, list):
                            processed_crop = processed_result[0]
                        else:
                            processed_crop = processed_result
                        
                        # Save the processed crop to temporary location
                        temp_modified_path = os.path.join(temp_dir, "modified", rel_path)
                        os.makedirs(os.path.dirname(temp_modified_path), exist_ok=True)
                        
                        if cv2.imwrite(temp_modified_path, processed_crop):
                            # Upload the modified crop to Google Storage
                            storage_modified_path = f"{modified_crops_folder}/{rel_path}"
                            
                            if self.tenant_storage.upload_from_file(storage_modified_path, temp_modified_path):
                                crops_processed += 1
                                logger.debug(f"Processed and uploaded crop: {storage_modified_path}")
                            else:
                                logger.warning(f"Failed to upload modified crop: {storage_modified_path}")
                        else:
                            logger.warning(f"Failed to save modified crop: {temp_modified_path}")
                            
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
                    "total_crops_downloaded": len(downloaded_crops)
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
        Augment player crops for training.
        
        Args:
            context: Pipeline context containing background removal results
            
        Returns:
            Dictionary with augmentation results
        """
        # Placeholder implementation
        background_removed = context.get("background_removed", False)
        modified_crops_folder = context.get("modified_crops_folder")
        crops_processed = context.get("crops_processed", 0)
        video_guid = context.get("video_guid")
        
        if background_removed and modified_crops_folder and crops_processed > 0:
            logger.info(f"Augmenting {crops_processed} crops for video: {video_guid}")
            logger.info(f"Using modified crops from: {modified_crops_folder}")
            return {"crops_augmented": True, "video_guid": video_guid, "modified_crops_folder": modified_crops_folder}
        else:
            logger.warning("No background-removed crops to augment")
            return {"crops_augmented": False}
    
    def _create_training_and_validation_sets(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create training and validation datasets from processed crops.
        
        Args:
            context: Pipeline context containing augmentation results
            
        Returns:
            Dictionary with dataset creation results
        """
        # Placeholder implementation
        crops_augmented = context.get("crops_augmented", False)
        video_guid = context.get("video_guid")
        if crops_augmented:
            logger.info(f"Creating training and validation sets for video: {video_guid}")
            return {"datasets_created": True, "video_guid": video_guid, "training_samples": 100}
        else:
            logger.warning("No augmented crops to create datasets from")
            return {"datasets_created": False}
    
    def _serialize_detections(self, detections: Detections) -> List[Dict[str, Any]]:
        """
        Serialize detections to JSON-compatible format.
        
        Args:
            detections: Supervision detections object
            
        Returns:
            List of serialized detection dictionaries
        """
        serialized = []
        
        for i in range(len(detections.xyxy)):
            detection = {
                "bbox": detections.xyxy[i].tolist(),
                "confidence": float(detections.confidence[i]) if detections.confidence is not None else 0.0,
                "class_id": int(detections.class_id[i]) if detections.class_id is not None else 0,
                "tracker_id": int(detections.tracker_id[i]) if detections.tracker_id is not None else None
            }
            serialized.append(detection)
        
        return serialized
    
    def _validate_video_resolution(self, video_path: str) -> bool:
        """
        Validate that video resolution is at least 1920 by 1080 pixels.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if resolution is adequate, False otherwise
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        is_valid = width >= 1920 and height >= 1080
        logger.info(f"Video resolution: {width}x{height}, Valid: {is_valid}")
        
        return is_valid
    
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
            logger.warning("No loaded video for frame extraction")
            return {"status": StepStatus.ERROR.value, "error": "No loaded video provided"}
        
        temp_path = loaded_video["temp_path"]
        video_guid = loaded_video["video_guid"]
        
        try:
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for frame extraction: {temp_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
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
                    frames_data.append(frame)
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

            logger.info(f"Extracted and saved {len(frames_data)} frames to {selected_frames_folder}")
            
            return {
                "frames_data": frames_data,
                "video_guid": video_guid,
                "frames_extracted": frame_ids,
                "selected_frames_folder": selected_frames_folder,
                "frames_saved_count": len(frames_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract frames from video {video_guid}: {e}")
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
            logger.warning("No frames data for grass mask initialization")
            return {"status": StepStatus.ERROR.value, "error": "No frames data provided"}
        
        try:
            logger.info(f"Initializing grass mask detector for video: {video_guid}")
            
            # Create frame generator from the extracted frames
            frame_generator = create_frame_generator_from_images(frames_data, input_format='BGR')
            
            # Initialize the background mask detector with the frames
            self.background_mask_detector.initialize(frame_generator)
            
            # Get background statistics for logging
            stats = self.background_mask_detector.get_stats()
            
            logger.info(f"Grass mask detector initialized successfully for video: {video_guid}")
            logger.debug(f"Background color statistics: {stats}")
            
            return {
                "grass_mask_initialized": True,
                "video_guid": video_guid,
                "background_stats": {
                    "mean_hsv": stats['mean_hsv'].tolist() if stats['mean_hsv'] is not None else None,
                    "std_hsv": stats['std_hsv'].tolist() if stats['std_hsv'] is not None else None,
                    "lower_bound": stats['lower_bound'].tolist() if stats['lower_bound'] is not None else None,
                    "upper_bound": stats['upper_bound'].tolist() if stats['upper_bound'] is not None else None,
                    "std_dev_multiplier": stats['std_dev_multiplier'],
                    "replacement_color": stats['replacement_color']
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize grass mask detector for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}


def run_training_pipeline(tenant_id: str = "tenant1", video_path: str = "", delete_original_raw_videos: bool = False, frames_per_video: int = 20, verbose: bool = False, save_intermediate: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run the training pipeline.
    
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
    pipeline = TrainPipeline(config, tenant_id=tenant_id, verbose=verbose, save_intermediate=save_intermediate)

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
    
    results = run_training_pipeline(tenant_id=tenant_id, video_path="/path/to/video.mp4", delete_original_raw_videos=delete_original_raw_videos, frames_per_video=frames_per_video, verbose=verbose, save_intermediate=save_intermediate)
    
    print("Pipeline Results:")
    print(json.dumps(results, indent=2))
