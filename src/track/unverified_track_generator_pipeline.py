"""
Track Generator Pipeline for LaxAI project.

This module provides a comprehensive pipeline for processing raw lacrosse video data into
unveri    Example:
        ```python
        from src.track.unverified_track_generator_pipeline import TrackGeneratorPipeline
        from config.all_config import detection_config

        # Initialize pipeline
        pipeline = TrackGeneratorPipeline(
            config=detection_config,
            tenant_id="lacrosse_team_1",
            verbose=True
        )

        # Process a video (relative GCS path, no gs:// prefix)
        results = pipeline.run("raw_videos/championship_game.mp4")

        if results["status"] == "completed":
            print(f"Successfully processed video {results['video_guid']}")
            print(f"Generated crops: {results['pipeline_summary']['total_crops']}")
        elif results["status"] == "cancelled":
            print(f"Pipeline was cancelled: {results['errors']}")

        # Stop pipeline programmatically
        pipeline.stop()

        # Check if stop was requested
        if pipeline.is_stopping():
            print("Pipeline stop has been requested")
        ```s. The pipeline handles video import, player detection, tracking,
crop generation, and organized storage for machine learning workflows.

Key Features:
    - Automated video processing with resolution validation (minimum 1920x1080)
    - Player detection using YOLO-based models with configurable confidence thresholds
    - Real-time tracking with affine transformation compensation
    - High-quality crop extraction with contrast and size filtering
    - Batch processing for memory efficiency with configurable concurrent uploads
    - Structured GCS storage organization with tenant-specific paths
    - Parallel processing for improved performance using ThreadPoolExecutor
    - Comprehensive error handling and progress tracking

Example:
    ```python
    from src.track.unverified_track_generator_pipeline import TrackGeneratorPipeline
    from config.all_config import detection_config

    # Initialize pipeline
    pipeline = TrackGeneratorPipeline(
        config=detection_config,
        tenant_id="lacrosse_team_1",
        verbose=True
    )

    # Process a video (relative GCS path, no gs:// prefix)
    results = pipeline.run("raw_videos/championship_game.mp4")

    if results["status"] == "completed":
        print(f"Successfully processed {results['video_guid']}")
        print(f"Generated {results['pipeline_summary']['total_crops']} player crops")
    elif results["status"] == "cancelled":
        print(f"Pipeline was cancelled: {results['errors']}")
    ```

Stop Pipeline Example:
    ```python
    from common.pipeline import stop_pipeline

    # Start pipeline
    pipeline = TrackGeneratorPipeline(config, "tenant1")
    results = pipeline.run("video.mp4")

    # Stop from another thread/process
    stop_pipeline("unverified_tracks_pipeline")

    # Or stop using instance method
    pipeline.stop()
    ```
"""

import logging
import cv2
import json
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import supervision as sv
from supervision.utils.image import crop_image

from common.google_storage import get_storage, GCSPaths
from common.detection import DetectionModel
from common.pipeline_step import StepStatus
from common.pipeline import Pipeline, PipelineStatus
from config.all_config import DetectionConfig, detection_config, model_config, training_config
from utils.id_generator import create_video_id, create_run_id
from common.tracker import AffineAwareByteTrack
from common.track_to_player import map_detections_to_players

logger = logging.getLogger(__name__)

#: Minimum video resolution required for processing (width, height)
MIN_VIDEO_RESOLUTION = (1920, 1080)
FRAME_SAMPLING_FOR_CROP = 15

#: Batch size for crop processing and upload (reduce memory usage for long videos)
CROP_BATCH_SIZE = 50

#: Maximum concurrent upload tasks to prevent overwhelming storage service
MAX_CONCURRENT_UPLOADS = 3

#: Interval for checkpoint saves (every N frames)
CHECKPOINT_FRAME_INTERVAL = 100


class TrackGeneratorPipeline(Pipeline):
    """
    Pipeline for generating unverified player tracks from lacrosse videos.

    This pipeline processes MP4 videos from Google Cloud Storage to detect players,
    track them across frames, and extract high-quality crops for training datasets.
    The pipeline uses batch processing for memory efficiency and parallel uploads
    for improved performance.

    **Pipeline Stages:**
        1. **Video Import**: Validate and import videos from tenant's raw directory
        2. **Detection & Tracking**: Run player detection on all frames with real-time tracking
        3. **Crop Extraction**: Extract player crops with quality filtering and batch processing

    **Key Features:**
        - Validates video resolution (minimum 1920x1080) during import
        - Player detection using YOLO-based models with confidence filtering
        - Real-time tracking with affine transformation compensation
        - High-quality crop extraction with contrast and size filtering
        - Batch processing for memory efficiency (configurable batch sizes)
        - Parallel upload operations with configurable concurrency limits
        - Structured GCS path organization with tenant-specific isolation
        - Comprehensive error handling and logging with progress tracking

    **Storage Organization:**
        ```
        tenant_id/
        ├── raw/{video_id}/
        ├── imported_videos/{video_id}/{video_id}.mp4
        ├── detections/{video_id}/detections.json
        └── unverified_tracks/{video_id}/{track_id}/crop_{detection_id}_{frame_id}.jpg
        ```

    Args:
        config (DetectionConfig): Detection configuration object containing model settings,
            GCS bucket information, and processing parameters
        tenant_id (str): The tenant ID for data organization and access control
        verbose (bool): Enable detailed logging throughout the pipeline
        save_intermediate (bool): Save intermediate results for debugging
        enable_grass_mask (bool): Enable background removal functionality (currently unused)
        delete_process_folder (bool): Clean up temporary processing files after completion
        **kwargs: Additional keyword arguments passed to parent Pipeline class

    Attributes:
        config (DetectionConfig): Detection configuration with model and processing parameters
        tenant_id (str): Tenant identifier for storage organization and data isolation
        enable_grass_mask (bool): Whether background removal is enabled (currently unused)
        detection_model (DetectionModel): Loaded detection model instance for player detection
        tracker (AffineAwareByteTrack): Real-time tracker with affine transformation compensation
        tenant_storage (GoogleStorageClient): GCS client for tenant-specific operations
        path_manager (GCSPaths): Structured path management system for GCS organization

    Raises:
        RuntimeError: If detection model or tracker fails to initialize
        ValueError: If invalid configuration parameters are provided

    Example:
        ```python
        from src.track.unverified_track_generator_pipeline import TrackGeneratorPipeline
        from config.all_config import detection_config

        # Initialize pipeline
        pipeline = TrackGeneratorPipeline(
            config=detection_config,
            tenant_id="lacrosse_team_1",
            verbose=True
        )

        # Process a video (relative GCS path, no gs:// prefix)
        results = pipeline.run("raw_videos/championship_game.mp4")

        if results["status"] == "completed":
            print(f"Successfully processed {results['video_guid']}")
            print(f"Generated crops for {results['pipeline_summary']['total_detections']} detections")
        ```

    Note:
        The pipeline requires a properly configured detection model and valid GCS credentials.
        Videos must meet minimum resolution requirements (1920x1080) for processing.
        All processing is tenant-isolated for multi-tenant environments.
    """
    
    # Crop quality constants
    MIN_CROP_WIDTH = 48
    MIN_CROP_HEIGHT = 96
    MIN_CROP_CONTRAST = 20.0

    def __init__(self, 
                 config: DetectionConfig, 
                 tenant_id: str, 
                 verbose: bool = True, 
                 save_intermediate: bool = True, 
                 enable_grass_mask: bool = model_config.enable_grass_mask, 
                 delete_process_folder: bool = True, 
                 **kwargs):
        """
        Initialize the TrackGeneratorPipeline.
        
        Sets up storage clients, detection models, tracker, and pipeline configuration.
        Configures the processing steps for player detection and tracking.
        
        Args:
            config (DetectionConfig): Detection configuration object containing model settings,
                GCS bucket information, and processing parameters
            tenant_id (str): The tenant ID for data organization and access control in multi-tenant environments
            verbose (bool): Enable detailed logging throughout the pipeline for monitoring progress
            save_intermediate (bool): Save intermediate results for debugging and recovery capabilities
            enable_grass_mask (bool): Enable background removal functionality using statistical color analysis.
                Currently unused in this pipeline
            delete_process_folder (bool): Clean up temporary processing files after completion to save storage
            **kwargs: Additional arguments passed to the parent Pipeline class
        
        Raises:
            RuntimeError: If the detection model or tracker fails to initialize
            ValueError: If tenant_id is empty or config contains invalid parameters
        """
        self.config = config
        self.tenant_id = tenant_id
        self.video_capture = None
        self.delete_process_folder = delete_process_folder
        self.dataloader_workers = training_config.num_workers
        
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
                                   context_info: str = "") -> Tuple[List, int, List[str]]:
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
            Tuple[List, int, List[str]]: Containing:
                - List of failed operations (same format as input tasks)
                - Integer count of successful operations
                - List of successful blob paths (for upload operations)
                
        Note:
            Uses self.dataloader_workers to limit concurrency and prevent overwhelming
            the storage service. Failed operations are logged but don't stop execution.
        """
        if not tasks:
            return [], 0, []

        # Check for cancellation request
        if self.is_stop_requested():
            logger.info(f"Stop requested, cancelling parallel {operation_func.__name__} operations")
            return [], 0, []

        logger.info(f"Starting parallel {operation_func.__name__} of {len(tasks)} items{' for ' + context_info if context_info else ''}")
        failed_operations = []
        successful_paths = []
        
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
                    if success:
                        # For upload tasks, extract the blob path (first element of tuple)
                        if isinstance(task, tuple) and len(task) > 0:
                            successful_paths.append(task[0])
                        elif isinstance(task, str):
                            successful_paths.append(task)
                    else:
                        failed_operations.append(task)
                        logger.error(f"Failed to run {operation_func.__name__}({task}) for {context_info}")
                except Exception as e:
                    failed_operations.append(task)
                    logger.error(f"Exception during {operation_func.__name__}({task}): {e}")

        successful_count = len(tasks) - len(failed_operations)
        
        if failed_operations:
            logger.warning(f"Failed to {operation_func.__name__} {len(failed_operations)} out of {len(tasks)} items{' for ' + context_info if context_info else ''}")

        logger.info(f"Completed {successful_count} {operation_func.__name__} operations{' for ' + context_info if context_info else ''}")

        return failed_operations, successful_count, successful_paths

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
                checkpoint data. If True, the pipeline will skip completed steps and resume
                from the exact frame where processing was interrupted.
                Defaults to True.
            
        Returns:
            Returns:
            Dictionary containing pipeline execution results:
                - status (str): Pipeline completion status ("completed", "error", "cancelled")
                - run_guid (str): Unique identifier for this pipeline run
                - run_folder (str): GCS path where run data is stored
                - video_path (str): Original input video path
                - video_guid (str): Generated unique video identifier
                - video_folder (str): GCS path where video data is organized
                - errors (List[str]): List of any errors encountered during processing
                - pipeline_summary (Dict): Summary statistics for each pipeline stage
                - resume_frame (int): Frame number where processing resumed from (if applicable)
                
        Raises:
            RuntimeError: If critical pipeline dependencies are missing
            ValueError: If video_path is invalid or empty
            
        Note:
            The pipeline can be stopped gracefully using the stop() method or stop_pipeline() function.
            When cancelled, the pipeline will save progress and return status "cancelled".
            Frame-level checkpoints enable resuming from exactly where processing stopped.
            
        Example:
            ```python
            pipeline = TrackGeneratorPipeline(detection_config, "tenant1")
            
            # Process video from raw uploads folder
            results = pipeline.run("raw_videos/game_footage.mp4")
            
            # Check if resumed from checkpoint
            if results.get("resume_frame"):
                print(f"Resumed processing from frame {results['resume_frame']}")
            
            if results["status"] == "completed":
                print(f"Successfully processed video {results['video_guid']}")
                print(f"Generated crops: {results['pipeline_summary']['total_crops']}")
            elif results["status"] == "cancelled":
                print(f"Pipeline was cancelled: {results['errors']}")
            
            # Stop pipeline programmatically
            pipeline.stop()
            ```
                
        Raises:
            RuntimeError: If critical pipeline dependencies are missing
            ValueError: If video_path is invalid or empty
            
        Example:
            ```python
            pipeline = TrackGeneratorPipeline(detection_config, "tenant1")
            
            # Process video from raw uploads folder
            results = pipeline.run("raw_videos/game_footage.mp4")
            
            # Process video from specific tenant folder
            results = pipeline.run("tenant1/uploads/championship_game.mp4")
            
            # Check results
            if results["status"] == "completed":
                print(f"Successfully processed video {results['video_guid']}")
                print(f"Generated crops: {results['pipeline_summary']['total_crops']}")
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
            "pipeline_summary": results["pipeline_summary"],
            "resume_frame": context.get("resume_frame")
        }

        # Cleanup temporary processing files if enabled
        if self.delete_process_folder:
            video_guid = context.get("video_guid", "unknown")
            process_folder_path = self.path_manager.get_path("process_folder", video_id=video_guid)
            logger.info(f"Cleaning up temporary process folder: {process_folder_path}")
            
            blob_names = self.tenant_storage.list_blobs(prefix=process_folder_path)
            if blob_names:
                failed_deletes, successful_deletes, _ = self._execute_parallel_operations(
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

    def stop(self) -> bool:
        """
        Stop this pipeline instance gracefully.

        This method requests the pipeline to stop execution at the next convenient point.
        The pipeline will complete any currently running step and save progress before stopping.

        Returns:
            bool: True if stop request was successfully initiated, False otherwise

        Example:
            ```python
            pipeline = TrackGeneratorPipeline(config, "tenant1")

            # Start pipeline in background
            import threading
            thread = threading.Thread(target=pipeline.run, args=("video.mp4",))
            thread.start()

            # Stop the pipeline
            pipeline.stop()
            ```
        """
        from common.pipeline import stop_pipeline
        return stop_pipeline(self.pipeline_name)

    def is_stopping(self) -> bool:
        """
        Check if a stop has been requested for this pipeline.

        Returns:
            bool: True if stop has been requested, False otherwise

        Note:
            This method is useful for custom logic that needs to check
            pipeline status without triggering cancellation.
        """
        return self.is_stop_requested()

    def _import_video(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a single video from the provided path into organized video folder.
        
        This method validates the video resolution, then moves the raw video file from its 
        original location to a structured GCS path within the tenant's storage area. It 
        generates a unique video ID and organizes the file according to the configured path structure.
        
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
            RuntimeError: If the video file cannot be moved to the target location or 
                         if video resolution doesn't meet minimum requirements (1920x1080)
        """
        raw_video_path = context.get("raw_video_path")
        if not raw_video_path:
            return {"status": StepStatus.ERROR.value, "error": "No video path provided"}

        try:
            logger.info(f"Importing video: {raw_video_path}")

            # Validate video resolution before importing
            with self.tenant_storage.get_video_capture(raw_video_path) as cap:
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video for resolution validation: {raw_video_path}")
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if not self._validate_video_resolution(width, height):
                    raise RuntimeError(f"Video resolution {width}x{height} does not meet minimum requirements of {MIN_VIDEO_RESOLUTION[0]}x{MIN_VIDEO_RESOLUTION[1]}")

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
            crop_tasks = []  # Collect async crop processing tasks
            current_batch = []  # Current batch of upload tasks
            upload_tasks = []  # Track concurrent upload tasks
            batch_counter = 0  # Track batch numbers for logging
            all_crop_paths = []  # Track all successful crop upload paths
            
            # Frame-level checkpoint configuration
            CHECKPOINT_FRAME_INTERVAL = 100  # Save checkpoint every 100 frames
            last_checkpoint_frame = -1
            
            # Check for frame-level resume information in context
            resume_frame = context.get("resume_frame", 0)
            resume_detections_count = context.get("resume_detections_count", 0)
            resume_all_detections = context.get("resume_all_detections", [])
            resume_crop_paths = context.get("resume_crop_paths", [])
            
            if resume_frame > 0:
                logger.info(f"Resuming video processing from frame {resume_frame}")
                detections_count = resume_detections_count
                all_detections = resume_all_detections
                all_crop_paths = resume_crop_paths
            
            with self.tenant_storage.get_video_capture(video_blob_name) as cap:
                if not cap.isOpened():
                    logger.error(f"Could not open video for detection: {video_blob_name}")
                    return {"status": StepStatus.ERROR.value, "error": f"Could not open video: {video_blob_name}"}
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"Processing {total_frames} frames for detection")
                
                # Seek to resume frame if resuming
                if resume_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
                    logger.info(f"Seeked to frame {resume_frame} for resume")
                
                frame_number = resume_frame if resume_frame > 0 else 0
                previous_frame_rgb = None
                while True:
                    # Check for cancellation request
                    if self.is_stop_requested():
                        logger.info(f"Stop requested during detection processing at frame {frame_number}/{total_frames}")
                        
                        # Process any remaining crops before stopping
                        if crop_tasks:
                            logger.info(f"Processing remaining {len(crop_tasks)} crop tasks before stopping...")
                            batch_paths = asyncio.run(self._process_crop_batch(crop_tasks, upload_tasks, batch_counter, video_guid))
                            if batch_paths:
                                all_crop_paths.extend(batch_paths)
                        
                        # Wait for any pending uploads to complete
                        if upload_tasks:
                            logger.info(f"Waiting for {len(upload_tasks)} pending upload tasks to complete before stopping...")
                            asyncio.run(asyncio.gather(*upload_tasks))
                        
                        # Log crops uploaded before cancellation
                        if all_crop_paths:
                            gcs_urls = [f"gs://{self.tenant_storage.bucket_name}/{path}" for path in all_crop_paths]
                            logger.info(f"Video {video_guid}: Pipeline stopped - {len(all_crop_paths)} crops uploaded before cancellation")
                            logger.info(f"Video {video_guid}: Crop URLs before cancellation - {', '.join(gcs_urls)}")
                        
                        # Return context with partial results
                        context.update({
                            "status": StepStatus.CANCELLED.value,
                            "all_detections": all_detections,
                            "total_detections": detections_count,
                            "crop_paths": all_crop_paths,
                            "total_crops": len(all_crop_paths),
                            "cancellation_reason": "Stop requested during detection processing",
                            "resume_frame": frame_number,  # Include current frame for resume
                            "resume_detections_count": detections_count,
                            "resume_all_detections": all_detections,
                            "resume_crop_paths": all_crop_paths,
                            "video_guid": video_guid,
                            "video_blob_name": video_blob_name,
                            "video_folder": video_folder
                        })
                        return context
                        
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
                            affine_matrix = self.tracker.calculate_affine_transform(previous_frame_rgb, frame_rgb)

                        # Apply the affine transformation to the detections
                        detections = self.tracker.update_with_transform(detections, affine_matrix, frame_rgb)

                        if frame_number % FRAME_SAMPLING_FOR_CROP == 0:
                            # Create async task for crop processing
                            task = asyncio.create_task(self._async_get_crops(frame_rgb.copy(), detections, video_guid, frame_number))
                            crop_tasks.append(task)
                            
                            # Check if we should process current batch
                            if len(crop_tasks) >= CROP_BATCH_SIZE:
                                batch_paths = asyncio.run(self._process_crop_batch(crop_tasks, upload_tasks, batch_counter, video_guid))
                                if batch_paths:
                                    all_crop_paths.extend(batch_paths)
                                crop_tasks = []  # Reset for next batch
                                batch_counter += 1

                    except Exception as e:
                        logger.error(f"Error processing frame {frame_number} for video {video_guid}: {e}")
                        detections = sv.Detections.empty()
                        
                    finally:
                        previous_frame_rgb = frame_rgb
                        # Set frame_index for all detections in this frame
                        if len(detections) > 0:
                            detections.data['frame_index'] = [frame_number] * len(detections)
                        else:
                            detections.data['frame_index'] = []
                        all_detections.append(detections)

                    frame_number += 1
                    
                    # Save frame-level checkpoint periodically
                    if frame_number % CHECKPOINT_FRAME_INTERVAL == 0 and frame_number != last_checkpoint_frame:
                        logger.debug(f"Saving checkpoint at frame {frame_number}")
                        
                        # Update context with current processing state
                        checkpoint_context = {
                            "resume_frame": frame_number,
                            "resume_detections_count": detections_count,
                            "resume_all_detections": all_detections,
                            "resume_crop_paths": all_crop_paths,
                            "video_guid": video_guid,
                            "video_blob_name": video_blob_name,
                            "video_folder": video_folder
                        }
                        
                        # Save checkpoint during step execution
                        # We need to get the current completed steps from the pipeline
                        current_completed_steps = self.current_completed_steps
                        if not self.save_checkpoint(checkpoint_context, current_completed_steps):
                            logger.warning(f"Failed to save checkpoint at frame {frame_number}")
                        else:
                            logger.debug(f"Checkpoint saved successfully at frame {frame_number}")
                        
                        last_checkpoint_frame = frame_number
                    
                    # Log progress every 100 frames
                    if frame_number % 100 == 0:
                        logger.info(f"Processed {frame_number}/{total_frames} frames, {detections_count} total detections")
            
            # Process any remaining crops in final batch
            if crop_tasks:
                logger.info(f"Processing final batch with {len(crop_tasks)} crop tasks...")
                batch_paths = asyncio.run(self._process_crop_batch(crop_tasks, upload_tasks, batch_counter, video_guid))
                if batch_paths:
                    all_crop_paths.extend(batch_paths)
                batch_counter += 1
            
            # Wait for all upload tasks to complete
            if upload_tasks:
                logger.info(f"Waiting for {len(upload_tasks)} upload batches to complete...")
                asyncio.run(asyncio.gather(*upload_tasks))
                logger.info("All crop upload batches completed")
            
            # Log all successful crop upload URLs
            if all_crop_paths:
                gcs_urls = [f"gs://{self.tenant_storage.bucket_name}/{path}" for path in all_crop_paths]
                logger.info(f"Video {video_guid}: Total crops uploaded: {len(all_crop_paths)}")
                logger.info(f"Video {video_guid}: All crop URLs - {', '.join(gcs_urls)}")
            else:
                logger.info(f"Video {video_guid}: No crops were uploaded")
            
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
                    'frame_index': list(merged_detections.data.get('frame_index', [])) if merged_detections.data and 'frame_index' in merged_detections.data else [],
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
                "crop_paths": all_crop_paths,
                "total_crops": len(all_crop_paths),
                # Clear resume information on successful completion
                "resume_frame": None,
                "resume_detections_count": None,
                "resume_all_detections": None,
                "resume_crop_paths": None,
            })
            
            return context
            
        except Exception as e:
            logger.error(f"Critical error in player detection for video {video_guid}: {e}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}

    async def _process_crop_batch(self, crop_tasks, upload_tasks, batch_counter, video_guid):
        """Process a batch of crop tasks and upload them concurrently.
        
        Returns:
            List[str]: List of successful blob paths that were uploaded.
        """
        if not crop_tasks:
            return []
            
        # Check for cancellation request
        if self.is_stop_requested():
            logger.info(f"Stop requested, skipping crop batch {batch_counter}")
            return []
            
        logger.info(f"Processing crop batch {batch_counter} with {len(crop_tasks)} tasks...")
        
        try:
            # Wait for crop processing tasks to complete
            all_upload_tasks = await asyncio.gather(*crop_tasks)
            
            # Flatten the list of upload tasks
            flat_upload_tasks = []
            for task_list in all_upload_tasks:
                if task_list:  # Check if task_list is not None
                    flat_upload_tasks.extend(task_list)
            
            if flat_upload_tasks:
                logger.info(f"Batch {batch_counter}: Uploading {len(flat_upload_tasks)} crops...")
                
                # Create upload task with concurrency control
                upload_task = asyncio.create_task(
                    self._upload_crop_batch(flat_upload_tasks, batch_counter, video_guid)
                )
                upload_tasks.append(upload_task)
                
                # Limit concurrent uploads to prevent overwhelming storage service
                if len(upload_tasks) >= MAX_CONCURRENT_UPLOADS:
                    logger.info(f"Reached max concurrent uploads ({MAX_CONCURRENT_UPLOADS}), waiting for some to complete...")
                    # Wait for the oldest upload task to complete and collect its successful paths
                    completed_task = await upload_tasks.pop(0)
                    logger.debug("Completed one upload batch, continuing...")
                    return completed_task if completed_task else []
            else:
                logger.debug(f"Batch {batch_counter}: No crops to upload")
                
        except Exception as e:
            logger.error(f"Error processing crop batch {batch_counter}: {e}")
            
        return []

    async def _upload_crop_batch(self, upload_tasks, batch_counter, video_guid):
        """Upload a batch of crops asynchronously.
        
        Returns:
            List[str]: List of successful blob paths that were uploaded.
        """
        # Check for cancellation request
        if self.is_stop_requested():
            logger.info(f"Stop requested, skipping upload batch {batch_counter}")
            return []
            
        try:
            failed_uploads, successful_uploads, successful_paths = await asyncio.get_event_loop().run_in_executor(
                None,
                self._execute_parallel_operations,
                upload_tasks,
                lambda task: self.tenant_storage.upload_from_bytes(
                    task[0], 
                    cv2.imencode('.jpg', cv2.cvtColor(task[1], cv2.COLOR_RGB2BGR))[1].tobytes()
                ),
                f"upload batch {batch_counter}"
            )
            
            logger.info(f"Batch {batch_counter}: Successfully uploaded {successful_uploads} crops")
            if failed_uploads:
                logger.warning(f"Batch {batch_counter}: Failed to upload {len(failed_uploads)} crops")
            
            # Log GCS URLs for successful uploads
            if successful_paths:
                gcs_urls = [f"gs://{self.tenant_storage.bucket_name}/{path}" for path in successful_paths]
                logger.info(f"Batch {batch_counter}: Crop URLs - {', '.join(gcs_urls)}")
            
            return successful_paths
                
        except Exception as e:
            logger.error(f"Error uploading crop batch {batch_counter}: {e}")
            return []

    async def _async_get_crops(self, frame, detections, video_guid: str, frame_number: int):
        """Async wrapper for crop processing to enable concurrent processing."""
        # Check for cancellation request
        if self.is_stop_requested():
            logger.debug(f"Stop requested, skipping crop processing for frame {frame_number}")
            return []
            
        try:
            logger.debug(f"Processing crops for frame {frame_number}")
            
            # Get upload tasks from synchronous method
            upload_tasks = self._get_crops(frame, detections, video_guid)
            
            return upload_tasks
            
        except Exception as e:
            logger.error(f"Error in async crop processing for frame {frame_number}: {e}")
            return []

    def _get_crops(self, frame, detections, video_guid: str):
        """
        Extract high-quality crops from frame detections and save to unverified tracks GCS path.
        
        This function processes detections from a frame, applies quality filters to discard
        low-quality crops, and saves the remaining crops to the appropriate GCS path structure.
        
        Args:
            frame: The frame image as numpy array
            detections: Supervision Detections object for the frame
            video_guid: Unique identifier for the video
            
        Returns:
            List[Tuple[str, np.ndarray]]: List of (blob_path, crop_image) tuples for upload
        """
        upload_tasks = []
        
        if not isinstance(detections, sv.Detections):
            logger.error("Invalid detections format - expected sv.Detections object")
            return upload_tasks

        # # Map detections to players using the new module
        # detections = map_detections_to_players(detections)

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
                    
                    # Apply quality filters
                    if not self._is_crop_quality_sufficient(crop_np):
                        logger.debug(f"Discarding low-quality crop for detection {det_idx} in track {track_id}")
                        continue
                    
                    # Get frame_id from detection data if available, otherwise use index
                    if detections.data and 'frame_index' in detections.data and isinstance(detections.data['frame_index'], (list, np.ndarray)):
                        frame_id = detections.data['frame_index'][det_idx] if det_idx < len(detections.data['frame_index']) else str(det_idx)
                    else:
                        frame_id = str(det_idx)
                    
                    # Generate GCS path for unverified tracks using the path manager
                    folder_path = self.path_manager.get_path("unverified_tracks", video_id=video_guid, track_id=track_id)
                    blob_path = f"{folder_path}crop_{det_idx}_{frame_id}.jpg"
                    
                    upload_tasks.append((blob_path, crop_np))
                    
                except Exception as e:
                    logger.error(f"Error extracting crop for detection {det_idx} in track {track_id}: {e}")
                    continue
        
        return upload_tasks

    def _is_crop_quality_sufficient(self, crop: np.ndarray) -> bool:
        """
        Check if a crop meets quality thresholds for reliable embeddings.
        
        Args:
            crop: The crop image as numpy array
            
        Returns:
            bool: True if crop quality is sufficient, False otherwise
        """
        # Check minimum size
        height, width = crop.shape[:2]
        if width < self.MIN_CROP_WIDTH or height < self.MIN_CROP_HEIGHT:
            logger.debug(f"Crop too small: {width}x{height} (min: {self.MIN_CROP_WIDTH}x{self.MIN_CROP_HEIGHT})")
            return False
        
        # Check contrast using standard deviation of grayscale image
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_crop)
        
        # Low contrast threshold
        if contrast < self.MIN_CROP_CONTRAST:
            logger.debug(f"Crop has low contrast: {contrast:.2f} < {self.MIN_CROP_CONTRAST}")
            return False
        
        return True


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
