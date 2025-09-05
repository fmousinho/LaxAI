"""
Data preparation pipeline for LaxAI project.

This module provides a comprehensive pipeline for processing raw lacrosse video data into
training-ready datasets. The pipeline handles video import, frame extraction, player detection,
crop generation, data augmentation, and dataset organization for machine learning workflows.

Key Features:
    - Automated video processing with resolution validation (minimum 1920x1080)
    - Player detection using YOLO-based models with configurable confidence thresholds
    - Optional background removal with grass mask detection and color analysis
    - Image augmentation for robust training data with configurable transforms
    - Parallel processing for improved performance using ThreadPoolExecutor
    - Structured GCS storage organization with tenant-specific paths
    - Train/validation dataset splitting with configurable ratios
    - Comprehensive error handling and progress tracking
    - Memory-efficient processing with configurable batch sizes

Example:
    ```python
    from src.train.dataprep_pipeline import DataPrepPipeline
    from config.all_config import detection_config

    # Initialize pipeline with background removal enabled
    pipeline = DataPrepPipeline(
        config=detection_config,
        tenant_id="lacrosse_team_1",
        verbose=True,
        enable_grass_mask=True,
        delete_process_folder=True
    )

    # Process a video (relative GCS path, no gs:// prefix)
    results = pipeline.run("raw_videos/championship_game.mp4")

    if results["status"] == "completed":
        print(f"Successfully processed {results['video_guid']}")
        print(f"Created datasets with {results['pipeline_summary']['total_train_samples']} training samples")
    ```
"""

import os
import logging
import random
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from common.tracker import AffineAwareByteTrack, TrackData
from common.track_to_player import map_detections_to_players

logger = logging.getLogger(__name__)

#: Minimum video resolution required for processing (width, height)
MIN_VIDEO_RESOLUTION = (1920, 1080)
FRAME_SAMPLING_FOR_CROP = 15


class DataPrepPipeline(Pipeline):
    """
    Comprehensive data preparation pipeline for lacrosse video processing.

    This pipeline processes MP4 videos from Google Cloud Storage through a series of stages
    to create training-ready datasets for player re-identification models. The pipeline is
    designed for scalability and robustness with parallel processing capabilities.

    **Pipeline Stages:**
        1. **Video Import**: Move videos from tenant's raw directory to organized structure
        2. **Frame Extraction**: Extract frames with sufficient quality for detection (equally spaced)
        3. **Grass Mask Calculation** (optional): Initialize background removal system with color analysis
        4. **Player Detection**: Detect players using YOLO-based models with confidence filtering
        5. **Crop Extraction**: Extract player bounding boxes as individual images with structured naming
        6. **Background Removal** (optional): Remove grass/field background from crops using mask detection
        7. **Data Augmentation**: Apply configurable transforms to increase dataset diversity
        8. **Dataset Creation**: Split into training and validation sets with proper GCS organization    **Key Features:**
        - Validates video resolution (minimum 1920x1080) and frame count
        - Parallel processing using ThreadPoolExecutor with configurable worker limits
        - Optional grass mask background removal with statistical color analysis
        - Structured GCS path organization with tenant-specific isolation
        - Comprehensive error handling and logging with detailed progress tracking
        - Memory-efficient processing with batch operations and cleanup
        - Automatic cleanup of temporary processing files
        - Resume capability for interrupted pipeline runs

    **Storage Organization:**
        ```
        tenant_id/
        ├── raw/{video_id}/
        ├── imported_videos/{video_id}/{video_id}.mp4
        ├── extracted_frames/{video_id}/{frame_id}/{frame_id}.jpg
        ├── crops/{video_id}/{frame_id}/{crop_id}.jpg
        ├── augmented_crops/{video_id}/{frame_id}/{orig_crop_id}/{aug_crop_id}.jpg
        └── datasets/{dataset_id}/train|val/{player_id}/
        ```

    Args:
        config (dict): Configuration dictionary containing:
            - 'gcs_bucket': GCS bucket name for data storage
            - 'detection_model_path': Path to YOLO detection model
            - 'min_resolution': Minimum video resolution (default: (1920, 1080))
            - 'max_workers': Maximum parallel workers for processing (default: 4)
            - 'batch_size': Batch size for frame processing (default: 32)
            - 'augmentation_config': Dictionary with augmentation parameters
            - 'tenant_id': Tenant identifier for multi-tenant isolation
        logger (logging.Logger, optional): Logger instance for tracking operations

    Attributes:
        config (DetectionConfig): Detection configuration object with model and processing parameters
        tenant_id (str): Tenant identifier for storage organization and data isolation
        frames_per_video (int): Number of frames to extract per video for processing
        train_ratio (float): Ratio of data used for training (vs validation) in dataset splits
        enable_grass_mask (bool): Whether background removal is enabled for crop processing
        detection_model (DetectionModel): Loaded YOLO detection model instance for player detection
        background_mask_detector (BackgroundMaskDetector, optional): Background removal system with statistical color analysis
        tenant_storage (GoogleStorageClient): GCS client for tenant-specific operations and data management
        path_manager (GCSPaths): Structured path management system for GCS organization and file naming

    Raises:
        RuntimeError: If detection model fails to load or GCS credentials are invalid
        ValueError: If invalid configuration parameters are provided or video resolution is insufficient
        FileNotFoundError: If specified video files or model files cannot be found
        ConnectionError: If GCS operations fail due to network or authentication issues

    Example:
        ```python
        from src.train.dataprep_pipeline import DataPrepPipeline
        from config.all_config import detection_config
        
        # Initialize with background removal enabled
        pipeline = DataPrepPipeline(
            config=detection_config,
            tenant_id="lacrosse_team_1",
            verbose=True,
            enable_grass_mask=True,
            delete_process_folder=True
        )
        
        # Process a game video (relative GCS path)
        results = pipeline.run("raw_videos/championship_game.mp4")
        
        if results["status"] == "completed":
            print(f"Successfully processed {results['video_guid']}")
            print(f"Created datasets with {results['pipeline_summary']['total_train_samples']} training samples")
        ```

    Note:
        The pipeline requires a properly configured detection model and valid GCS credentials.
        Videos must meet minimum resolution requirements (1920x1080) for processing.
        All processing is tenant-isolated for multi-tenant environments.
    """

    def __init__(self, 
                 config: DetectionConfig, 
                 tenant_id: str, 
                 verbose: bool = True, 
                 save_intermediate: bool = True, 
                 enable_grass_mask: bool = model_config.enable_grass_mask, 
                 delete_process_folder: bool = True, 
                 **kwargs):
        """
        Initialize the data preparation pipeline.
        
        Sets up storage clients, detection models, and pipeline configuration.
        Configures the processing steps based on grass mask settings and tenant isolation.
        
        Args:
            config (DetectionConfig): Detection configuration object containing model settings,
                GCS bucket information, and processing parameters
            tenant_id (str): The tenant ID for data organization and access control in multi-tenant environments
            verbose (bool): Enable detailed logging throughout the pipeline for monitoring progress
            save_intermediate (bool): Save intermediate results for debugging and recovery capabilities
            enable_grass_mask (bool): Enable background removal functionality using statistical color analysis.
                If None, uses transform_config.enable_background_removal setting
            delete_process_folder (bool): Clean up temporary processing files after completion to save storage
            **kwargs: Additional arguments passed to the parent Pipeline class
        
        Raises:
            RuntimeError: If the detection model fails to load, which is required for the pipeline to function
            ValueError: If tenant_id is empty or config contains invalid parameters
        """
        self.config = config
        self.tenant_id = tenant_id
        self.video_capture = None
        self.train_ratio = training_config.train_ratio
        self.delete_process_folder = delete_process_folder
        self.dataloader_workers = training_config.dataloader_workers
        
        # Import transform_config to get the background removal setting
        from config.all_config import transform_config
        
        # Get storage client
        self.tenant_storage = get_storage(tenant_id)  # For tenant-specific operations (without /user suffix)
        
        # Initialize GCS path manager for structured path handling
        self.path_manager = GCSPaths()
        
        # Store tenant_id for path generation
        self.tenant_id = tenant_id
        
        # Detection model is required for training pipeline
        try:
            self.detection_model = DetectionModel()
            logger.info("Detection model loaded successfully")
        except RuntimeError as e:
            logger.critical(f"CRITICAL ERROR: Detection model is required for training pipeline but failed to load: {e}")
            raise RuntimeError(f"Training pipeline cannot continue without detection model: {e}")
        
        try:
            self.tracker = AffineAwareByteTrack(False, 15)
            logger.info("Tracker initialized successfully")
        except RuntimeError as e:
            logger.critical(f"CRITICAL ERROR: Tracker is required for training pipeline but failed to initialize: {e}")
            raise RuntimeError(f"Training pipeline cannot continue without tracker: {e}")

        # Define base pipeline steps (always included)
        step_definitions = [
            ("import_videos", {
                "description": "Move MP4 videos from tenant's raw directory for processing",
                "function": self._import_video
            }),
            ("generate_detections", {
                "description": "Run player detection and tracking model on whole video",
                "function": self._get_detections_and_tracks
            })
            # ("extract_track_crops", {
            #     "description": "Extract crops for each track",
            #     "function": self._extract_track_crops
            # })
        ]

        step_definitions = dict(step_definitions)

        # Initialize base pipeline
        super().__init__(
            pipeline_name="unverified_tracks_pipeline",
            storage_client=self.tenant_storage,
            step_definitions=step_definitions,
            verbose=verbose,
            save_intermediate=save_intermediate
        )
        
        # Override run_folder to use structured GCS path
        self.run_guid = create_run_id()
        self.run_folder = self.path_manager.get_path("run_data", run_id=self.run_guid)

    def _execute_parallel_operations(self, 
                                   tasks: List[Tuple] | List[str], 
                                   operation_func, 
                                   context_info: str = "") -> Tuple[List, int]:
        """
        Execute a list of operations in parallel using ThreadPoolExecutor.
        
        This method provides a generic interface for parallelizing storage operations
        such as uploads, downloads, moves, and deletions. It handles error collection,
        provides comprehensive logging, and uses configurable worker limits.
        
        Args:
            tasks (List[Tuple] | List[str]): List of task parameters. Can be:
                - List of tuples for multi-parameter operations (e.g., (source, dest) for moves)
                - List of strings for single-parameter operations (e.g., blob names for deletes)
            operation_func: Function to execute for each task. Should return bool indicating success.
                Examples: self.tenant_storage.upload_from_bytes, self.tenant_storage.move_blob
            context_info (str): Additional context information for logging (e.g., "video_123 crops")
            
        Returns:
            Tuple[List, int]: Containing:
                - List of failed operations (same format as input tasks)
                - Integer count of successful operations
                
        Note:
            Uses self.dataloader_workers to limit concurrency and prevent overwhelming
            the storage service. Failed operations are logged but don't stop execution.
        """
        if not tasks:
            return [], 0

        logger.info(f"Starting parallel {operation_func.__name__} of {len(tasks)} items{' for ' + context_info if context_info else ''}")
        failed_operations = []
        
        with ThreadPoolExecutor(max_workers=self.dataloader_workers) as executor:
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

        logger.info(f"Completed {successful_count} {operation_func.__name__} operations{' for ' + context_info if context_info else ''}")

        return failed_operations, successful_count

    def run(self, video_path: str, resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Execute the complete data preparation pipeline for a single video.
        
        This is the main entry point for processing lacrosse videos stored in Google Cloud Storage.
        The method orchestrates all pipeline stages and provides comprehensive error handling
        and progress tracking.
        
        Args:
            video_path: Relative GCS blob path to the video file to process within the tenant's 
                storage bucket. Should NOT include the 'gs://' prefix or bucket name.
                Examples:
                - "raw_videos/game_footage.mp4"
                - "uploads/2024/championship_game.mp4" 
                - "tenant1/imported_videos/video_123.mp4"
            resume_from_checkpoint: Whether to check for and resume from existing 
                checkpoint data. If True, the pipeline will skip completed steps.
                Defaults to True.
            
        Returns:
            Dictionary containing pipeline execution results:
                - status (str): Pipeline completion status ("completed", "error", "partial")
                - run_guid (str): Unique identifier for this pipeline run
                - run_folder (str): GCS path where run data is stored
                - video_path (str): Original input video path
                - video_guid (str): Generated unique video identifier
                - video_folder (str): GCS path where video data is organized
                - errors (List[str]): List of any errors encountered during processing
                - pipeline_summary (Dict): Summary statistics for each pipeline stage
                
        Raises:
            RuntimeError: If critical pipeline dependencies are missing
            ValueError: If video_path is invalid or empty
            
        Example:
            ```python
            pipeline = DataPrepPipeline(detection_config, "tenant1")
            
            # Process video from raw uploads folder
            results = pipeline.run("raw_videos/game_footage.mp4")
            
            # Process video from specific tenant folder
            results = pipeline.run("tenant1/uploads/championship_game.mp4")
            
            # Check results
            if results["status"] == "completed":
                print(f"Successfully processed video {results['video_guid']}")
                print(f"Training samples: {results['pipeline_summary']['total_train_samples']}")
            else:
                print(f"Pipeline failed: {results['errors']}")
            ```
            
        Note:
            - All video files must already be uploaded to Google Cloud Storage
            - Video paths are relative to the configured GCS bucket
            - The pipeline automatically handles temporary file cleanup if 
              delete_process_folder is enabled
            - All intermediate data is stored in structured GCS paths for easy 
              organization and retrieval
        """
        if not video_path:
            return {"status": PipelineStatus.ERROR.value, "error": "No video path provided"}

        logger.info(f"Starting data preparation pipeline for video: {video_path}")

        # Create initial context with the video path
        context = {"raw_video_path": video_path}
        
        # Call the base class run method with the initial context
        # The base class now handles checkpoint functionality automatically
        results = super().run(context, resume_from_checkpoint=resume_from_checkpoint)
        
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

        # Cleanup temporary processing files if enabled
        if self.delete_process_folder:
            video_guid = context.get("video_guid", "unknown")
            process_folder_path = self.path_manager.get_path("process_folder", video_id=video_guid)
            logger.info(f"Cleaning up temporary process folder: {process_folder_path}")
            
            blob_names = self.tenant_storage.list_blobs(prefix=process_folder_path)
            if blob_names:
                failed_deletes, successful_deletes = self._execute_parallel_operations(
                    list(blob_names),
                    self.tenant_storage.delete_blob,
                    f"cleanup process folder {process_folder_path}"
                )
                
                if failed_deletes:
                    logger.warning(f"Failed to delete {len(failed_deletes)} temporary files from {process_folder_path}")
                
                logger.info(f"Successfully cleaned up {successful_deletes} temporary files from {process_folder_path}")
            else:
                logger.info("No temporary files found to clean up")

        return formatted_results

    def _import_video(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a single video from the provided path into organized video folder.
        
        This method moves the raw video file from its original location to a structured
        GCS path within the tenant's storage area. It generates a unique video ID and
        organizes the file according to the configured path structure.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - raw_video_path: The original GCS path of the video file
                
        Returns:
            Dict[str, Any]: Updated context with import results:
                - status: Step completion status
                - video_guid: Generated unique identifier for the video
                - video_folder: Structured GCS folder path for the video
                - video_blob_name: Full GCS path to the imported video file
                - error: Error message if import failed
                
        Raises:
            RuntimeError: If the video file cannot be moved to the target location
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
    
    
    def _get_detections_and_tracks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run player detection on all frames in the video using the model.
        
        This method processes every frame in the video through the detection model to identify
        players. It uses supervision library for proper detection format and handles batch processing
        with confidence filtering.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - video_blob_name: GCS path to the imported video file
                - video_guid: Unique identifier for the video being processed
                - video_folder: GCS folder path for the video
                
        Returns:
            Dict[str, Any]: Updated context with detection results:
                - status: Step completion status
                - detections_data: List of detection results for each frame
                - total_detections: Total number of player detections found
                - error: Error message if detection failed
                
        Note:
            Detection results are saved to GCS in structured paths for use by
            subsequent crop extraction steps. Empty detections are logged as warnings
            but don't fail the pipeline.
        """
        video_blob_name = context.get("video_blob_name")
        video_guid = context.get("video_guid")  
        video_folder = context.get("video_folder")

        if not self.detection_model:
            logger.error("No detection model available for player detection")
            return {"status": StepStatus.ERROR.value, "error": "Detection model not initialized"}
        
        if not self.tracker:
            logger.error("No tracker available for player detection")
            return {"status": StepStatus.ERROR.value, "error": "Tracker not initialized"}

        if not video_blob_name:
            logger.error("No video blob name for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No video blob name provided"}
        
        if not video_guid:
            logger.error("No video GUID for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No video GUID provided"}
        
        if not video_folder:
            logger.error("No video folder for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No video folder provided for saving detections"}
        
        try:
            logger.info(f"Starting player detection for video: {video_guid}")
            
            # Process all frames in the video
            detections_count = 0
            all_detections = []
            
            with self.tenant_storage.get_video_capture(video_blob_name) as cap:
                if not cap.isOpened():
                    logger.error(f"Could not open video for detection: {video_blob_name}")
                    return {"status": StepStatus.ERROR.value, "error": f"Could not open video: {video_blob_name}"}
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"Processing {total_frames} frames for detection")
                
                frame_number = 0
                previous_frame_rgb = None
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    try:
                        # Convert BGR to RGB for model input
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Generate detections using the model
                        detections = self.detection_model.generate_detections(frame_rgb)
                        
                        if detections is None or len(detections) == 0:
                            logger.debug(f"Frame {frame_number}: No detections found")
                            detections = sv.Detections.empty()
                        else:
                            detections_count += len(detections)
                            logger.debug(f"Frame {frame_number}: Found {len(detections)} detections")

                        if frame_number == 0:
                            affine_matrix = self.tracker.get_identity_affine_matrix()
                        else:
                            affine_matrix = self.tracker.get_affine_matrix(previous_frame_rgb, frame_rgb)

                        # Apply the affine transformation to the detections
                        detections = self.tracker.track(detections, affine_matrix)

                    except Exception as e:
                        logger.error(f"Error processing frame {frame_number} for video {video_guid}: {e}")
                        detections = sv.Detections.empty()
                        
                    finally:
                        previous_frame_rgb = frame_rgb
                        detections.data['frame_index'] = frame_number
                        all_detections.append(detections)

                    frame_number += 1
                    
                    # Log progress every 100 frames
                    if frame_number % 100 == 0:
                        logger.info(f"Processed {frame_number}/{total_frames} frames, {detections_count} total detections")
            
            # Save detections - merge all frame detections
            detections_blob_name = f"{video_folder.rstrip('/')}/detections.json"
            
            if all_detections:
                # Merge all detections from all frames
                merged_detections = sv.Detections.merge(all_detections)
                # Convert to JSON-serializable format for storage
                detections_data = {
                    'xyxy': merged_detections.xyxy.tolist() if merged_detections.xyxy is not None else [],
                    'confidence': merged_detections.confidence.tolist() if merged_detections.confidence is not None else [],
                    'class_id': merged_detections.class_id.tolist() if merged_detections.class_id is not None else [],
                    'frame_index': merged_detections.data.get('frame_index', []).tolist() if merged_detections.data and 'frame_index' in merged_detections.data else [],
                    'total_frames': len(all_detections),
                    'total_detections': detections_count
                }
                
            else:
                logger.warning(f"No detections found for video {video_guid}")
                detections_data = {
                    'xyxy': [],
                    'confidence': [],
                    'class_id': [],
                    'total_frames': 0,
                    'total_detections': 0
                }
            
            self.tenant_storage.upload_from_bytes(detections_blob_name, json.dumps(detections_data).encode('utf-8'))

            logger.info(f"Player detection completed for video {video_guid} - {detections_count} detections found across {len(all_detections)} frames")

            if detections_count == 0:
                logger.error(f"No detections found for video {video_guid}")
                return {"status": StepStatus.ERROR.value, "error": "No detections found in video frames"}

            context.update({
                "status": StepStatus.COMPLETED.value,
                "all_detections": all_detections,
                "total_detections": detections_count,
            })
            
            return context
            
        except Exception as e:
            logger.error(f"Critical error in player detection for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}


    def _map_players(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map player IDs to their corresponding track IDs in the context.
        
        This method is deprecated and no longer used. Player mapping is now handled
        by the track_to_player module's map_detections_to_players function.
        """
        # This method is kept for backwards compatibility but is no longer used
        # Player mapping is now handled in _extract_crops_from_frame using map_detections_to_players
        return {"status": StepStatus.COMPLETED.value}
        """
        Generate the folder path for crops based on track_id and frame_id.
        
        Args:
            track_id: The track identifier for the detection
            frame_id: The frame identifier
            video_guid: The video identifier
            
        Returns:
            str: The folder path for storing crops
        """
        # Create a unique folder name combining track_id and frame_id
        folder_name = f"track_{track_id}_frame_{frame_id}"
        return f"{video_guid}/crops/{folder_name}"


    def _extract_crops_from_frame(self, frame: np.ndarray, video_folder: str, detections: sv.Detections, video_guid: str) -> List[Tuple[str, np.ndarray]]:
        """
        Extract and prepare crops from a single frame's detections.
        
        This helper function processes detections from a single frame and organizes
        crops by track_id. Each track gets its own folder structure.
        
        Args:
            frame: The frame image as numpy array
            video_folder: Base video folder path
            detections: Supervision Detections object for the frame
            video_guid: Video identifier for path generation
            
        Returns:
            List[Tuple[str, np.ndarray]]: List of (blob_path, crop_image) tuples for upload
        """
        upload_tasks = []
        players_in_frame = set()
        
        if not isinstance(detections, sv.Detections):
            logger.error("Invalid detections format - expected sv.Detections object")
            return upload_tasks

        if not detections.data or 'detections' not in detections.data:
            logger.error("No detections found in Supervision Detections object")
            return upload_tasks
        


        # Map detections to players using the new module
        detections = map_detections_to_players(detections)

        # Get player IDs for this frame
        player_ids = detections.data.get('player_id', [])
        players_in_frame = set(player_ids)

        track_detections = {}
        for i, detection in enumerate(detections):
            track_id = detection[4] if len(detection) > 4 else 0  # tracker_id is at index 4
            if track_id not in track_detections:
                track_detections[track_id] = []
            track_detections[track_id].append((i, detection))
        
        # Process each track's detections
        for track_id, track_det_list in track_detections.items():
            for det_idx, detection in track_det_list:
                try:
                    # Extract crop from frame
                    crop_np = crop_image(frame, detection[0].astype(int))
                    
                    # Get frame_id from detection data if available, otherwise use index
                    frame_id = detections.data.get('frame_index', [det_idx])[det_idx] if detections.data and 'frame_index' in detections.data else str(det_idx)
                    
                    # Generate folder path for this track
                    crop_folder = self._get_crop_folder_path(track_id, str(frame_id), video_guid)
                    
                    # Create unique filename for this crop
                    file_name = f"crop_{det_idx}.jpg"
                    blob_path = f"{crop_folder}/{file_name}"
                    
                    upload_tasks.append((blob_path, crop_np))
                    
                except Exception as e:
                    logger.error(f"Error extracting crop for detection {det_idx} in track {track_id}: {e}")
                    continue
        
        return upload_tasks
    

    def _extract_track_crops(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract crops from all frames in the video using the detections.
        
        This pipeline step processes every frame in the video, extracts player crops
        based on the detection results, and uploads them to GCS with structured paths.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - video_blob_name: GCS path to the imported video file
                - video_guid: Unique identifier for the video being processed
                - video_folder: GCS folder path for the video
                - all_detections: List of detection results for each frame
                
        Returns:
            Dict[str, Any]: Updated context with crop extraction results:
                - status: Step completion status
                - total_crops_extracted: Total number of crops extracted
                - error: Error message if extraction failed
                
        Note:
            Crops are saved to GCS in structured paths organized by track_id and frame_id.
            Each crop includes player mapping information from the track_to_player module.
        """
        video_blob_name = context.get("video_blob_name")
        video_guid = context.get("video_guid") 
        video_folder = context.get("video_folder")
        all_detections = context.get("all_detections", [])

        if not video_blob_name:
            logger.error("No video blob name for crop extraction")
            return {"status": StepStatus.ERROR.value, "error": "No video blob name provided"}
        
        if not video_guid:
            logger.error("No video GUID for crop extraction")
            return {"status": StepStatus.ERROR.value, "error": "No video GUID provided"}
        
        if not video_folder:
            logger.error("No video folder for crop extraction")
            return {"status": StepStatus.ERROR.value, "error": "No video folder provided for saving crops"}
        
        if not all_detections:
            logger.error("No detections available for crop extraction")
            return {"status": StepStatus.ERROR.value, "error": "No detections found for crop extraction"}

        try:
            logger.info(f"Starting crop extraction for video: {video_guid}")
            
            total_crops = 0
            frame_number = 0
            
            with self.tenant_storage.get_video_capture(video_blob_name) as cap:
                if not cap.isOpened():
                    logger.error(f"Could not open video for crop extraction: {video_blob_name}")
                    return {"status": StepStatus.ERROR.value, "error": f"Could not open video: {video_blob_name}"}
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"Processing {total_frames} frames for crop extraction")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    try:
                        # Convert BGR to RGB for processing
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Get detections for this frame
                        if frame_number < len(all_detections):
                            detections = all_detections[frame_number]
                        else:
                            logger.warning(f"No detections found for frame {frame_number}")
                            detections = sv.Detections.empty()
                        
                        # Extract crops from this frame
                        upload_tasks = self._extract_crops_from_frame(frame_rgb, video_folder, detections, video_guid)
                        
                        # Upload crops in parallel
                        if upload_tasks:
                            failed_uploads, successful_uploads = self._execute_parallel_operations(
                                upload_tasks,
                                lambda task: self.tenant_storage.upload_from_bytes(task[0], task[1].tobytes()),
                                f"upload crops for frame {frame_number}"
                            )
                            
                            total_crops += successful_uploads
                            
                            if failed_uploads:
                                logger.warning(f"Failed to upload {len(failed_uploads)} crops for frame {frame_number}")
                        
                        # Log progress every 100 frames
                        if frame_number % 100 == 0:
                            logger.info(f"Processed {frame_number}/{total_frames} frames, {total_crops} total crops extracted")
                            
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_number} for crop extraction in video {video_guid}: {e}")
                        continue
                        
                    finally:
                        frame_number += 1
            
            logger.info(f"Crop extraction completed for video {video_guid} - {total_crops} crops extracted")

            context.update({
                "status": StepStatus.COMPLETED.value,
                "total_crops_extracted": total_crops,
            })
            
            return context
            
        except Exception as e:
            logger.error(f"Critical error in crop extraction for video {video_guid}: {e}")
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


    
            

    def _initialize_grass_mask(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the grass mask detector using the extracted frames.
        
        This method analyzes the extracted frames to identify background colors
        and initialize the background removal system. It performs statistical
        analysis of frame colors to create a mask for grass/field removal.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - frames_data: List of extracted frame images as numpy arrays
                - video_guid: Unique identifier for the video being processed
                
        Returns:
            Dict[str, Any]: Updated context with initialization results:
                - status: Step completion status
                - grass_mask_initialized: Boolean indicating successful initialization
                - background_stats: Statistical information about detected background
                - error: Error message if initialization failed
                
        Note:
            This step is conditionally executed based on enable_grass_mask configuration.
            Background statistics are logged for debugging and monitoring purposes.
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

            logger.info(f"Grass mask detector initialized for video: {video_guid}")
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


