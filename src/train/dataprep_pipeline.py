"""
Data preparation pipeline for LaxAI project.

This module defines the DataPrepPipeline class and related utilities for processing raw video data,
including downloading from Google Storage, extracting frames, augmenting images, and preparing datasets for training.
"""


import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import random
import supervision as sv
from PIL import Image
from supervision import Detections
from supervision.utils.image import crop_image
from common.google_storage import get_storage, GCSPaths
from common.detection import DetectionModel
from common.pipeline_step import StepStatus
from common.pipeline import Pipeline, PipelineStatus
from config.all_config import DetectionConfig, detection_config, model_config, training_config
from config import logging_config
from common.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
from train.augmentation import augment_images
from utils.id_generator import create_video_id, create_frame_id, create_dataset_id, create_run_id, create_crop_id, create_aug_crop_id

logger = logging.getLogger(__name__)

MIN_VIDEO_RESOLUTION = (1920, 1080)  # Minimum resolution for video processing


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

    def __init__(self, config: DetectionConfig, tenant_id: str = "tenant1", verbose: bool = True, save_intermediate: bool = True, enable_grass_mask: bool = model_config.enable_grass_mask, delete_process_folder: bool = True, **kwargs):
        """
        Initialize the training pipeline.
        
        Args:
            config: Detection configuration object
            tenant_id: The tenant ID to process videos for (default: "tenant1")
            verbose: Enable verbose logging (default: False)
            save_intermediate: Save intermediate results for each step (default: False)
            enable_grass_mask: Enable or disable grass mask functionality. If None, uses transform_config.enable_background_removal (default: None)
        """
        self.config = config
        self.tenant_id = tenant_id
        self.frames_per_video = config.frames_per_video
        self.video_capture = None
        self.train_ratio = training_config.train_ratio
        self.delete_process_folder = delete_process_folder
        self.default_workers = training_config.default_workers
        
        # Import transform_config to get the background removal setting
        from config.all_config import transform_config
        
        # Determine if grass mask should be enabled
        if enable_grass_mask is None:
            self.enable_grass_mask = transform_config.enable_background_removal
        else:
            self.enable_grass_mask = enable_grass_mask
            
        # Only initialize background mask detector if grass mask is enabled
        if self.enable_grass_mask:
            self.background_mask_detector = BackgroundMaskDetector()
            logger.info("Grass mask functionality enabled - BackgroundMaskDetector initialized")
        else:
            self.background_mask_detector = None
            logger.info("Grass mask functionality disabled - skipping BackgroundMaskDetector initialization")
        
        # Get storage client
        self.tenant_storage = get_storage(tenant_id)  # For tenant-specific operations (without /user suffix)
        
        # Initialize GCS path manager for structured path handling
        self.path_manager = GCSPaths()
        
        # Store tenant_id for path generation
        self.tenant_id = tenant_id
        
        # Detection model is required for training pipeline
        try:
            self.detection_model = DetectionModel()
            logger.info("Detection model successfully loaded")
        except RuntimeError as e:
            logger.critical(f"CRITICAL ERROR: Detection model is required for training pipeline but failed to load: {e}")
            raise RuntimeError(f"Training pipeline cannot continue without detection model: {e}")
        
        # Define base pipeline steps (always included)
        base_steps = [
            ("import_videos", {
                "description": "Move MP4 videos from tenant's raw directory for processing",
                "function": self._import_video
            }),
            ("extract_frames", {
                "description": "Extract frames with sufficient detections for processing",
                "function": self._extract_frames_for_detections
            })
        ]
        
        # Conditionally add grass mask step
        if self.enable_grass_mask:
            base_steps.append(("calculate_grass_mask", {
                "description": "Calculate grass mask for each frame",
                "function": self._initialize_grass_mask
            }))
            
        # Continue with detection and crop extraction
        base_steps.extend([
            ("detect_players", {
                "description": "Process detection results and save to storage",
                "function": self._detect_players
            }),
            ("extract_crops", {
                "description": "Extract and save player crops from detections",
                "function": self._extract_crops
            })
        ])
        
        # Conditionally add background removal step
        if self.enable_grass_mask:
            base_steps.append(("remove_crop_background", {
                "description": "Remove background from player crops",
                "function": self._remove_crop_background
            }))
            
        # Continue with augmentation and dataset creation
        base_steps.extend([
            ("augment_crops", {
                "description": "Augment player crops for training",
                "function": self._augment_crops
            }),
            ("create_training_and_validation_sets", {
                "description": "Create training and validation datasets from processed crops",
                "function": self._create_training_and_validation_sets
            })
        ])
        
        # Convert to dictionary while preserving order
        step_definitions = dict(base_steps)
        
        if self.enable_grass_mask:
            logger.info("Added grass mask and background removal steps to pipeline")
        else:
            logger.info("Skipped grass mask and background removal steps (disabled by configuration)")

        # Initialize base pipeline
        super().__init__(
            pipeline_name="train_pipeline",
            storage_client=self.tenant_storage,
            step_definitions=step_definitions,
            verbose=verbose,
            save_intermediate=save_intermediate
        )
        
        # Override run_folder to use structured GCS path
        self.run_guid = create_run_id()
        self.run_folder = self.path_manager.get_path("run_data", run_id=self.run_guid)


    def _execute_parallel_operations(self, tasks: List[Tuple] | List[str], operation_func, context_info: str = "") -> Tuple[List, int]:
        """
        Execute a list of operations in parallel using ThreadPoolExecutor.
        
        Args:
            tasks: List of tuples containing task parameters
            operation_func: Function to execute for each task (e.g., self.tenant_storage.upload_from_bytes)
            context_info: Additional context information for logging
            
        Returns:
            Tuple of (failed_operations, successful_count)
        """
        if not tasks:
            return [], 0

        logger.info(f"Starting parallel {operation_func.__name__} of {len(tasks)} items{' for ' + context_info if context_info else ''}")
        failed_operations = []
        
        with ThreadPoolExecutor(max_workers=self.default_workers) as executor:
            # Submit tasks using the provided function
            # For upload: operation_func(*task) - blob_name, data  
            # For move: operation_func(*task) - source, destination
            # For delete: operation_func(task) - blob_name (string)
            future_to_task = {
                executor.submit(operation_func, *task) if isinstance(task, tuple) else executor.submit(operation_func, task): task
                for task in tasks
            }
            
            # Process completed operations
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    success = future.result()
                    if not success:
                        failed_operations.append(task)
                        logger.error(f"Failed to run {operation_func.__name__}({task}) for {context_info}")
                except Exception as e:
                    failed_operations.append(task)
                    logger.error(f"Exception during {operation_func.__name__}({task}): {e}")

        successful_count = len(tasks) - len(failed_operations)
        
        if failed_operations:
            logger.warning(f"Failed to {operation_func.__name__} {len(failed_operations)} out of {len(tasks)} items{' for ' + context_info if context_info else ''}")

        logger.info(f"Successfully completed {successful_count} {operation_func.__name__} operations{' for ' + context_info if context_info else ''}")

        return failed_operations, successful_count


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
        initial_context = {"raw_video_path": video_path}
        
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

        if self.delete_process_folder:
            video_guid = context.get("video_guid", "unknown")
            process_folder_path = self.path_manager.get_path("process_folder", video_id=video_guid)
            logger.info(f"Deleting process folder: {process_folder_path}")
            blob_names = self.tenant_storage.list_blobs(prefix=process_folder_path)
            # Convert blob names to tuple format for delete operation (blob_name, None)
            failed_deletes, deletes = self._execute_parallel_operations(
                blob_names,
                self.tenant_storage.delete_blob,
                f"delete process folder {process_folder_path}"
            )
            if failed_deletes:
                logger.warning(f"Failed to delete {len(failed_deletes)} from {process_folder_path}")
            logger.info(f"Successfully deleted {deletes} from {process_folder_path}")

        return formatted_results
    
    
    def _import_video(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a single video from the provided path into organized video folder.
        
        Args:
            context: Pipeline context containing video_path
            
        Returns:
            Dictionary with imported video information
        """
        raw_video_path = context.get("raw_video_path")
        if not raw_video_path:
            return {"status": StepStatus.ERROR.value, "error": "No video path provided"}

        try:
            logger.info(f"Importing video: {raw_video_path}")

            # Generate structured video ID using ID generator
            video_guid = create_video_id()
            
            video_folder = self.path_manager.get_path("imported_video", 
                                                    video_id=video_guid).rstrip('/')
            
            imported_video_blob_name = f"{video_folder}/{video_guid}.mp4"


            if not self.tenant_storage.move_blob(raw_video_path, imported_video_blob_name):
                raise RuntimeError(f"Failed to move video from {raw_video_path} to {imported_video_blob_name}")
       
            logger.info(f"Successfully imported to {imported_video_blob_name}")

            context.update({
                "status": StepStatus.COMPLETED.value,
                "video_guid": video_guid,
                "video_folder": video_folder,
                "video_blob_name": imported_video_blob_name
            })

            return context

        except Exception as e:
            logger.error(f"Failed to import video {video_path}: {str(e)}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    
    
    def _detect_players(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run player detection on the extracted frames.
        
        Args:
            context: Pipeline context containing frames_data
            
        Returns:
            Dictionary with detection results
        """
        frames_data = context.get("frames_data")
        frame_ids = context.get("frame_ids")
        video_guid = context.get("video_guid")  
        video_folder = context.get("video_folder")
        
        if not frames_data:
            logger.error("No frames data for player detection - previous step failed or frames unavailable")
            return {"status": StepStatus.ERROR.value, "error": "No frames data provided - frame extraction failed"}
        
        if not video_guid:
            logger.error("No video GUID for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No video GUID provided"}
        
        if not video_folder:
            logger.error("No video folder for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No video folder provided for saving detections"}
        
        if type(frame_ids) is not list or len(frame_ids) != len(frames_data):
            logger.error("Frame IDs do not match frames data length")
            return {"status": StepStatus.ERROR.value, "error": "Frame IDs and frames data length mismatch"}

        try:
            logger.info(f"Starting player detection for video: {video_guid} with {len(frames_data)} frames")
            
            # Process each frame individually since the detection model expects individual frames
            detections_count = 0
            all_detections = []
            for frame_id, frame in zip(frame_ids, frames_data):
           
                try:     
                # Generate detections for this single frame
                    frame_detections = self.detection_model.generate_detections(frame)
                    if frame_detections is None or len(frame_detections) == 0:
                        logger.warning(f"Frame {frame_id}: No detections found")
                    all_detections.append(frame_detections)
                    detections_count += len(frame_detections)

                except Exception as e:
                    logger.error(f"Error processing frame {frame_id} for video {video_guid}: {e}")
                    raise RuntimeError(f"Failed to process frame {frame_id} for video {video_guid}: {e}")
                        

            # Save detections - convert to JSON-serializable format
            detections_blob_name = f"{video_folder.rstrip('/')}/detections.json"
            single_detection_object = sv.Detections.merge(all_detections)

            self.tenant_storage.upload_from_bytes(detections_blob_name, single_detection_object)

            logger.info(f"Player detection completed for frame id {frame_id} - {detections_count} detections found")

            if detections_count == 0:
                logger.warning(f"No detections found for video {video_guid} - skipping crop extraction")
                return {"status": StepStatus.ERROR.value, "error": "No detections found in video frames"}

            context.update({
                "status": StepStatus.COMPLETED.value,
                "all_detections": all_detections
            })
            
            return context
            
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
        all_detections = context.get("all_detections")
        frames_data = context.get("frames_data")
        video_guid = context.get("video_guid") 
        frame_ids = context.get("frame_ids")
        frames_guids = context.get("frames_guids")

        if not all_detections or not frames_data:
            logger.warning("No detection result or frames data found for crop extraction")
            return {"status": StepStatus.ERROR.value, "error": "No detection result or frames data provided"}
        
        try:
            logger.info(f"Extracting crops for video: {video_guid}")

            crops_by_frame = []
            upload_tasks = []
            
            # First phase: Extract all crops and collect upload tasks
            for frame_detections, frame, frame_id, frame_guid in zip(all_detections, frames_data, frame_ids, frames_guids):
                crops = []
                if not isinstance(frame_detections, sv.Detections):
                    logger.error("Invalid detections format - expected sv.Detections object")
                    raise RuntimeError("Invalid detections format - expected sv.Detections object")
                
                for detection in frame_detections:
                    crop_np = crop_image(frame, detection[0].astype(int))
                    crop_pil = Image.fromarray(crop_np, mode='RGB')

                    file_name = create_crop_id(frame_id=frame_guid)
                    blob_path = self.path_manager.get_path("orig_crops", 
                                                           video_id=video_guid, 
                                                           frame_id=frame_guid)
                    
                    blob_name = f"{blob_path.rstrip('/')}/{file_name}"

                    # Add to upload tasks instead of uploading immediately
                    upload_tasks.append((blob_name, crop_np))
                    crops.append(crop_np)
                
                crops_by_frame.append(crops)

            # Second phase: Execute uploads in parallel
            failed_uploads, crops_uploaded = self._execute_parallel_operations(
                upload_tasks, 
                self.tenant_storage.upload_from_bytes,
                f"crops for video {video_guid}"
            )
            
            if failed_uploads:
                logger.warning(f"Failed to upload {len(failed_uploads)} out of {len(upload_tasks)} crops for {video_guid}")
            
            logger.info(f"Successfully extracted and uploaded {crops_uploaded} crops across {len(frames_data)} frames")
            
            context.update({
                "status": StepStatus.COMPLETED.value,
                "crops_by_frame": crops_by_frame
            })

            return context
                        
        except Exception as e:
            logger.error(f"Failed to extract crops for  {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    

    def _remove_crop_background(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove background from player crops using the initialized grass mask detector.
        Replaces the original crop with modified crop in memory. Does not upload to storage.
        
        Args:
            context: Pipeline context containing crop extraction results
            
        Returns:
            Dictionary with background removal results
        """
       
        if not self.enable_grass_mask:
            logger.info("Grass mask disabled - skipping background removal step")
            return {"status": StepStatus.ERROR.value, "error": "Background removal step called but grass mask is disabled"}
        
        crops_by_frame = context.get("crops_by_frame")
        video_guid = context.get("video_guid")

        if not context.get("grass_mask_initialized"):
            logger.error("Grass mask detector not initialized")
            return {"status": StepStatus.ERROR.value, "error": "Grass mask detector not initialized"}
        
        crops_processed = 0

        try:
            logger.info(f"Removing background from crops for video: {video_guid}")
            for frame in crops_by_frame:
                for crop in frame:
                      processed_crop = self.background_mask_detector.remove_background(
                                crop_img, 
                                input_format='RGB'
                            )
                      crop = processed_crop.copy()
                      crops_processed += 1
            context.update({
                "status": StepStatus.COMPLETED.value,
                "crops_wo_background": crops_processed
            })
            logger.info(f"Successfully removed background from {crops_processed} crops for video {video_guid}")
            return context

                        
        except Exception as e:
            logger.error(f"Failed to remove background from crops for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    

    def _augment_crops(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augment player crops for training and store them in structured GCS paths per frame.
        
        Args:
            context: Pipeline context containing crop processing results
            
        Returns:
            Dictionary with augmentation results
        """
        video_guid = context.get("video_guid")
        frames_guid = context.get("frames_guids")
        crops_by_frame = context.get("crops_by_frame")

        
        logger.info(f"Augmenting crops for {video_guid}")

        n_aug_crops = 0
        
        try:
            # Collect all upload tasks first
            upload_tasks = []
            
            for crops, frames_guid in zip(crops_by_frame, frames_guid):
                for crop in crops:
                    #creates augmented crops for each original crop
                    aug_crops = augment_images([crop])
                    temp_crop_guid = create_crop_id(frame_id=frames_guid).rstrip('.jpg')  # Remove .jpg extension for folder name
                    blob_path = self.path_manager.get_path("augmented_crops",
                                                               video_id=video_guid, 
                                                               frame_id=frames_guid,
                                                               orig_crop_id=temp_crop_guid)
                    for aug_crop in aug_crops:
                        file_name = create_aug_crop_id(temp_crop_guid, n_aug_crops)
                        blob_name = f"{blob_path.rstrip('/')}/{file_name}"
                        
                        # Add to upload tasks instead of uploading immediately
                        upload_tasks.append((blob_name, aug_crop))
                        n_aug_crops += 1

            # Execute uploads in parallel
            failed_uploads, successful_uploads = self._execute_parallel_operations(
                upload_tasks, 
                self.tenant_storage.upload_from_bytes,
                f"augmented crops for video {video_guid}"
            )

            # Check for upload failures
            if failed_uploads:
                logger.error(f"Failed to upload {len(failed_uploads)} out of {n_aug_crops} augmented crops")
                return {"status": StepStatus.ERROR.value, "error": f"Failed to upload {len(failed_uploads)} augmented crops"}

            logger.info(f"Successfully created and uploaded {n_aug_crops} augmented crops for {video_guid}")
            context.update({
                "status": StepStatus.COMPLETED.value,
                "crops_augmented": True,
                "n_augmented_crops": n_aug_crops
            })
            return context
        
        except Exception as e:
            logger.error(f"Failed to augment crops for {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    
    
    def _create_training_and_validation_sets(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create training and validation datasets from processed crops.
        Each frame gets its own dataset with separate train/val splits.
        
        Args:
            context: Pipeline context containing augmentation results
            
        Returns:
            Dictionary with dataset creation results
        """
        crops_augmented = context.get("crops_augmented", False)
        frames_guids = context.get("frames_guids")
        video_guid = context.get("video_guid")
        
        if not crops_augmented:
            logger.warning("No augmented crops to create datasets from")
            return {"status": StepStatus.ERROR.value, "error": "No augmented crops available"}
        
        try:
            logger.info(f"Creating training and validation sets for video {video_guid}")

            for frame_guid in frames_guids:

                aug_crops_root = self.path_manager.get_path("augmented_crops_root", video_id=video_guid, frame_id=frame_guid).rstrip('/')
                if not aug_crops_root:
                    logger.error(f"No augmented crops root found for video {video_guid} at frame {frame_guid}")
                    raise RuntimeError(f"No augmented crops root found for video {video_guid} at frame {frame_guid}")
                players = self.tenant_storage.list_blobs(prefix=aug_crops_root, delimiter='/')
                players = list(players)  
                random.shuffle(players)  
                
                dataset_guid = create_dataset_id()
                n_train_players = int(len(players) * self.train_ratio)
                train_players = players[:n_train_players]
                val_players = players[n_train_players:]
                

                ttl_train=0
                ttl_val=0

                train_folder = self.path_manager.get_path("train_dataset", dataset_id=dataset_guid).rstrip('/')
                val_folder = self.path_manager.get_path("val_dataset", dataset_id=dataset_guid).rstrip('/')


                move_tasks = []
                for player in train_players:
                    player_folder = self.path_manager.get_path("augmented_crops", video_id = video_guid, dataset_id=dataset_guid, orig_crop_id=player.rstrip('/'))
                    player_images_path_list = self.tenant_storage.list_blobs(prefix=player_folder)
                    destination_folder = f"{train_folder}/{player}"
                    destination_folder = destination_folder.rstrip('/')

                    for image_path in player_images_path_list:
                        if not image_path.endswith('.jpg'):
                            logger.warning(f"Skipping non-JPG file: {image_path}")
                            continue
                        move_tasks.append((image_path, f"{destination_folder}/{os.path.basename(image_path)}"))

                for player in val_players:
                    player_folder = self.path_manager.get_path("augmented_crops", video_id = video_guid, dataset_id=dataset_guid, orig_crop_id=player.rstrip('/'))
                    player_images_path_list = self.tenant_storage.list_blobs(prefix=player_folder)
                    destination_folder = f"{val_folder}/{player}"
                    destination_folder = destination_folder.rstrip('/')

                    for image_path in player_images_path_list:
                        if not image_path.endswith('.jpg'):
                            logger.warning(f"Skipping non-JPG file: {image_path}")
                            continue
                        move_tasks.append((image_path, f"{destination_folder}/{os.path.basename(image_path)}"))

                failed_moves, successful_moves = self._execute_parallel_operations(
                    move_tasks, 
                    self.tenant_storage.move_blob,
                    f"crops to train/val datasets for {frame_guid}"
                )

                # Count successful moves by type
                ttl_train = 0
                ttl_val = 0
                for task in move_tasks:
                    if task not in failed_moves:
                        if task[1].split('/')[-3] == "train":  # dataset_type is the 3rd element
                            ttl_train += 1
                        else:
                            ttl_val += 1

                if failed_moves:
                    logger.warning(f"Failed to move {len(failed_moves)} out of {len(move_tasks)} crops for {frame_guid}")
                
                logger.info(f"Created {dataset_guid} with {ttl_train} training and {ttl_val} val crops for {frame_guid}")

            context.update({
                "status": StepStatus.COMPLETED.value,
                "datasets_created": True,
                "total_train_samples": ttl_train,
                "total_val_samples": ttl_val
            })
            return context

        except Exception as e:
            logger.error(f"Failed to create training and validation sets for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}


    def _validate_video_resolution(self, width: int, height: int) -> bool:
        """
        Validate that video resolution complies with minimum requirements.

        Args:
            width: Width of the video, in pixels
            height: Height of the video, in pixels

        Returns:
            True if resolution is adequate, False otherwise
        """

        return (width >= MIN_VIDEO_RESOLUTION[0] and height >= MIN_VIDEO_RESOLUTION[1])


    def _extract_video_metadata(self, cap) -> Dict[str, Any]:
        """
        Extract metadata from video file.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            Dictionary with video metadata
        """
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for metadata extraction: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration
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
        video_blob_name = context.get("video_blob_name", None)
        video_guid = context.get("video_guid", None)
        
        
        if not video_blob_name:
            logger.error("No loaded video for frame extraction - previous step failed")
            return {"status": StepStatus.ERROR.value, "error": "No  video found - video loading failed"}
       
        
        with self.tenant_storage.get_video_capture(video_blob_name) as cap:

            if not cap.isOpened():
                logger.error(f"Could not open video for frame extraction: {video_blob_name}")
                return {"status": StepStatus.ERROR.value, "error": f"Could not open video: {video_blob_name}"}

            video_metadata = self._extract_video_metadata(cap)

            if not self._validate_video_resolution(video_metadata["width"], video_metadata["height"]):
                logger.error(f"Video resolution {video_metadata['width']}x{video_metadata['height']} is below minimum requirement")
                return {"status": StepStatus.ERROR.value, "error": "Video resolution is below minimum requirements"}

            total_frames = video_metadata["frame_count"]
            
            if total_frames <= 0:
                logger.error(f"Video has no frames or invalid frame count: {total_frames}")
                return {"status": StepStatus.ERROR.value, "error": f"Invalid frame count: {total_frames}"}
            
            # Calculate initial frame positions (frames_per_video frames equally spaced)
            frame_positions = [int(i * total_frames / self.frames_per_video) for i in range(self.frames_per_video)]
            
            frames_data = []
            frames_guids = []
            extracted_frame_paths = {}
            
            logger.debug(f"Starting frame extraction for {len(frame_positions)} positions")
            
            for frame_position in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                logger.debug(f"Processing frame {frame_position}")
                    # Generate structured frame ID
                frame_guid = create_frame_id()

                
                frames_data.append(frame)
                frames_guids.append(frame_guid)
                
                # Create structured path for extracted frames
                frame_folder = self.path_manager.get_path("extracted_frames", 
                                                        video_id=video_guid, 
                                                        frame_id=frame_guid)
                
                # Save frame to Google Storage using structured path
                frame_filename = f"{frame_guid}.jpg"
                frame_blob_path = f"{frame_folder.rstrip('/')}/{frame_filename}"


                success = self.tenant_storage.upload_from_bytes(frame_blob_path, frame)
                if not success:
                    logger.error(f"Failed to upload {frame_guid} to storage path: {frame_blob_path}")
                    return {"status": StepStatus.ERROR.value, "error": f"Failed to upload frame {frame_guid} to storage path: {frame_blob_path}"}
           
                    

            if len(frames_data) == 0:
                raise RuntimeError(f"No frames could be extracted from video")

            logger.info(f"Extracted and saved {len(frames_data)} frames with structured IDs")

            context.update({
                "status": StepStatus.COMPLETED.value,
                "frames_data": frames_data,
                "video_guid": video_guid,
                "frames_guids": frames_guids,
                "frame_ids": frame_positions
            })
            
            return context
            

    def _initialize_grass_mask(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the grass mask detector using the extracted frames.
        
        Args:
            context: Pipeline context containing frames_data
            
        Returns:
            Dictionary with grass mask initialization results
        """
        # Check if grass mask is enabled
        if not self.enable_grass_mask:
            logger.info("Grass mask disabled - skipping grass mask initialization step")
            return {"status": StepStatus.ERROR.value, "error": "Grass mask initialization step called but grass mask is disabled"}
        
        frames_data = context.get("frames_data")
        video_guid = context.get("video_guid")
        
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

            context.update({
                "status": StepStatus.COMPLETED.value,
                "grass_mask_initialized": True,
                "background_stats": background_stats
            })
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to initialize grass mask detector for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}


def run_dataprep_pipeline(*args, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run the data preparation pipeline.

    Args:
        tenant_id: The tenant ID to process videos for
        video_path: Path to the video file to process
        delete_original_raw_videos: Whether to delete original raw video files after processing (default: False)
        frames_per_video: Number of frames to extract per video
        verbose: Whether to enable verbose logging (default: False)
        save_intermediate: Whether to save intermediate results (default: False)
        
    Returns:
        Dictionary with pipeline results
    """
    config = DetectionConfig()
    config.delete_original_raw_videos = delete_original_raw_videos
    config.frames_per_video = frames_per_video
    pipeline = DataPrepPipeline(config, *args, **kwargs)

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
