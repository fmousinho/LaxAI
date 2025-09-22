"""
Track Generator Pipeline for LaxAI project.

This module provides a comprehensive pipeline for processing raw lacrosse video data into
unveri    Example:
        ```python
        from unverified_track_generator_pipeline import TrackGeneratorPipeline
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
    from unverified_track_generator_pipeline import TrackGeneratorPipeline
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
from matplotlib.style import context
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
        from unverified_track_generator_pipeline import TrackGeneratorPipeline
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

    async def _execute_parallel_operations_async(self,
                                               tasks: List[Tuple] | List[str],
                                               operation_func,
                                               context_info: str = "") -> Tuple[List, int]:
        """
        Execute a list of operations in parallel using asyncio.

        This method provides a generic interface for parallelizing storage operations
        such as uploads, downloads, moves, and deletions. It handles error collection,
        provides comprehensive logging, and uses configurable concurrency limits.

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

        Note:
            Uses asyncio.gather to limit concurrency and prevent overwhelming
            the storage service. Failed operations are logged but don't stop execution.
        """
        if not tasks:
            return [], 0

        logger.info(f"Starting parallel {operation_func.__name__} of {len(tasks)} items{' for ' + context_info if context_info else ''}")
        failed_operations = []

        # Create async tasks for each operation
        async def execute_single_task(task):
            try:
                # Run the synchronous operation in a thread pool
                loop = asyncio.get_event_loop()
                if isinstance(task, tuple):
                    success = await loop.run_in_executor(None, operation_func, *task)
                else:
                    success = await loop.run_in_executor(None, operation_func, task)

                if success:
                   return True, task
                else:
                    return False, task  # (None, failed_task)
            except Exception as e:
                logger.error(f"Exception during {operation_func.__name__}({task}): {e}")
                return False, task  # (None, failed_task)

        # Execute all tasks concurrently, but limit concurrency
        semaphore = asyncio.Semaphore(self.dataloader_workers)

        async def execute_with_semaphore(task):
            async with semaphore:
                return await execute_single_task(task)

        # Run all tasks
        task_coroutines = [execute_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                failed_operations.append(tasks[i])
                continue
            elif result is None or not isinstance(result, tuple):
                logger.error(f"Task returned None result for {tasks[i]}")
                failed_operations.append(tasks[i])
                continue
            elif result[0] is False:
                failed_operations.append(tasks[i])

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
                checkpoint data. If True, the pipeline will skip completed steps and resume
                from the exact frame where processing was interrupted.
                Defaults to True.
            
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
                failed_deletes, successful_deletes = asyncio.run(self._execute_parallel_operations_async(
                    list(blob_names),
                    self.tenant_storage.delete_blob,
                    f"cleanup process folder {process_folder_path}"
                ))
                
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
        Import video from raw path to structured GCS location.
        
        Validates resolution and moves video to organized tenant folder.
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
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                
                if not self._validate_video_resolution(width, height):
                    raise RuntimeError(f"Video resolution {width}x{height} does not meet minimum requirements of {MIN_VIDEO_RESOLUTION[0]}x{MIN_VIDEO_RESOLUTION[1]}")

            # Generate structured video ID using ID generator
            video_guid = create_video_id()
            
            path = self.path_manager.get_path("imported_video", 
                                                    video_id=video_guid)
            if path:
                video_folder = path.rstrip('/')
            else:
                raise RuntimeError("Unable to determine video path")
            
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
            logger.error(f"Failed to import video {raw_video_path}: {str(e)}")
            return {"status": StepStatus.ERROR.value, "error": str(e)}
    
    def _get_detections_and_tracks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run player detection and tracking on video frames.
        
        Processes all frames with detection model and tracker, extracts crops asynchronously.
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
            get_crop_tasks = []  # Collect async crop processing tasks
            upload_tasks = []  # Track concurrent upload tasks
            batch_counter = 0  # Track batch numbers for logging
            all_crop_paths = []  # Track all successful crop upload paths
           
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
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                logger.info(f"Processing {total_frames} frames for detection")
                
                # Seek to resume frame if resuming
                if resume_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
                    logger.info(f"Seeked to frame {resume_frame} for resume")
                
                frame_number = resume_frame if resume_frame > 0 else 0
                previous_frame_rgb: np.ndarray
                while True:
                    # Check for cancellation request
                    if self.is_stop_requested():
                        logger.info(f"Stop requested during detection processing at frame {frame_number}/{total_frames}")
                        update = self._graceful_stop(
                            get_crop_tasks, upload_tasks, batch_counter, video_guid, 
                            all_crop_paths, frame_number, detections_count, all_detections,
                            video_blob_name, video_folder
                        )
                        context.update(update)
                        return context
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    try:
                        # Convert BGR to RGB for model input
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #type: ignore
                        
                        detections = self.detection_model.generate_detections(frame_rgb)
                        if type(detections) is not sv.Detections:
                            raise RuntimeError("Unrecognized detection type")
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
                            task = asyncio.create_task(self._get_upload_list_for_frame(frame_rgb.copy(), detections, video_guid, frame_number))
                            get_crop_tasks.append(task)

                            if len(get_crop_tasks) >= CROP_BATCH_SIZE:
                                task = asyncio.create_task(self._upload_crop_batch(get_crop_tasks, batch_counter))
                                upload_tasks.append(task)
                                get_crop_tasks = []  # Reset for next batch
                                batch_counter += 1

                    except Exception as e:
                        logger.error(f"Error processing frame {frame_number} for video {video_guid}: {e}")
                        detections = sv.Detections.empty()
                        
                    finally:
                        previous_frame_rgb = frame_rgb
                        # Set frame_index for all detections in this frame
                        if type(detections) is not sv.Detections:
                            raise RuntimeError("Unrecognized detection type")
                        elif len(detections) > 0:
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
            if len(get_crop_tasks) > 0:
                final_upload_task = asyncio.create_task(self._upload_crop_batch(get_crop_tasks, batch_counter))
                upload_tasks.append(final_upload_task)
                batch_counter += 1
            
            # Wait for all upload tasks to complete and collect results
            if upload_tasks:
                logger.info(f"Waiting for {len(upload_tasks)} upload batches to complete...")
                
                # Wait for all upload batch tasks to complete
                async def wait_for_uploads():
                    await asyncio.gather(*upload_tasks)
                
                asyncio.run(wait_for_uploads())
                
                logger.info(f"All crop upload batches completed")
            
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

    async def _upload_crop_batch(self, get_crop_tasks: List[asyncio.Task], batch_counter: int):
        """Upload a batch of crops asynchronously.
        
        Returns:
            List[str]: List of successful blob paths that were uploaded.
        """
            
        try:
            upload_tasks = await asyncio.gather(*get_crop_tasks)

            failed_uploads, successful_uploads = await self._execute_parallel_operations_async(
                upload_tasks,
                lambda task: self.tenant_storage.upload_from_bytes(
                    task[0], 
                    cv2.imencode('.jpg', cv2.cvtColor(task[1], cv2.COLOR_RGB2BGR))[1].tobytes()
                ),
                f"upload batch {batch_counter}"
            )
            
            logger.debug(f"Batch {batch_counter}: Successfully uploaded {successful_uploads} crops")
            if failed_uploads:
                logger.error(f"Batch {batch_counter}: Failed to upload {len(failed_uploads)} crops") 
            return
                
        except Exception as e:
            logger.error(f"Error uploading crop batch {batch_counter}: {e}")
            return

    def _get_crops(self, frame: np.ndarray, detections: sv.Detections) -> List[Tuple[int, np.ndarray]]:
        """
        Extract quality-filtered crops from frame detections.
        
        Returns list of (track_id, crop_image) tuples for valid detections.
        """
        res = []
        for detection in detections:
            crop = crop_image(frame, detection[0].astype(int))
            if self._is_crop_quality_sufficient(crop):
                crop_tuple = (detection[4], crop)  # (track_id, crop)
                res.append(crop_tuple)
        return res
        # upload_tasks = []

        # # # Map detections to players using the new module
        # # detections = map_detections_to_players(detections)

        # track_detections = {}
        # for i, detection in enumerate(detections):
        #     track_id = detection[4] if len(detection) > 4 else 0  # tracker_id is at index 4
        #     if track_id not in track_detections:
        #         track_detections[track_id] = []
        #     track_detections[track_id].append((i, detection))
        
        # # Process each track's detections
        # for track_id, track_det_list in track_detections.items():
        #     for det_idx, detection in track_det_list:
        #         try:
        #             # Extract crop from frame
        #             crop_np = crop_image(frame, detection[0].astype(int))
                    
        #             # Apply quality filters
        #             if not self._is_crop_quality_sufficient(crop_np):
        #                 logger.debug(f"Discarding low-quality crop for detection {det_idx} in track {track_id}")
        #                 continue
                    
        #             # Get frame_id from detection data if available, otherwise use index
        #             if detections.data and 'frame_index' in detections.data and isinstance(detections.data['frame_index'], (list, np.ndarray)):
        #                 frame_id = detections.data['frame_index'][det_idx] if det_idx < len(detections.data['frame_index']) else str(det_idx)
        #             else:
        #                 frame_id = str(det_idx)
                    
        #             # Generate GCS path for unverified tracks using the path manager
        #             folder_path = self.path_manager.get_path("unverified_tracks", video_id=video_guid, track_id=track_id)
        #             blob_path = f"{folder_path}crop_{det_idx}_{frame_id}.jpg"
                    
        #             upload_tasks.append((blob_path, crop_np))
                    
        #         except Exception as e:
        #             logger.error(f"Error extracting crop for detection {det_idx} in track {track_id}: {e}")
        #             continue
        
        # return upload_tasks

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
        contrast = np.std(gray_crop)  #type: ignore
        
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

    async def _get_upload_list_for_frame(
            self, frame: np.ndarray, 
            detections: sv.Detections, 
            video_guid: str, 
            frame_number: int) -> List[Tuple[str, np.ndarray]]:
        """
        Generate upload tasks for crops from a single frame.
        
        Returns list of (blob_path, crop_image) tuples for batch upload.
        """
        crop_tuples = self._get_crops(frame, detections)
        upload_list = []
        if len(crop_tuples) > 0:
            for crop_tuple in crop_tuples:
                crop_dir = self.path_manager.get_path("unverified_tracks", video_id=video_guid, track_id=crop_tuple[0])
                if crop_dir is None:
                    raise RuntimeError(f"Unable to determine crop directory for {video_guid}, track {crop_tuple[0]}")
                crop_dir = crop_dir.rstrip('/') + '/'
                crop_path = f"{crop_dir}crop_{frame_number}.jpg"
                upload_list_tuple = (crop_path, crop_tuple[1])
                upload_list.append(upload_list_tuple)

            logger.debug(f"Generated upload list for frame {frame_number}: {upload_list}")
        else:
            logger.warning(f"No valid crops found for frame {frame_number}")
            return upload_list

        return upload_list

    def _graceful_stop(self, crop_tasks, upload_tasks, batch_counter, video_guid, 
                      all_crop_paths, frame_number, detections_count, all_detections,
                      video_blob_name, video_folder) -> Dict[str, Any]:
        """
        Handle graceful pipeline shutdown during video processing.

        Completes all pending crop extraction and upload tasks before stopping,
        preventing data loss and enabling resume capability.

        Args:
            crop_tasks: Pending crop extraction tasks to complete
            upload_tasks: Active upload batch tasks in progress
            batch_counter: Current batch counter for logging
            video_guid: Unique video identifier
            all_crop_paths: Successfully uploaded crop paths
            frame_number: Current frame where processing stopped
            detections_count: Total detections found so far
            all_detections: All detection results collected
            video_blob_name: GCS path to video file
            video_folder: GCS folder for video data

        Returns:
            Dict containing cancellation context with partial results for resume,
            including status, detections, crop paths, and resume information.

        Note:
            Ensures no data loss by completing pending operations before cancellation.
        """
        logger.info("Initiating graceful stop of the pipeline...")
        
        if crop_tasks:
            logger.info(f"Processing remaining {len(crop_tasks)} crop tasks before stopping...")
            # Create a final upload batch task for remaining crops
            final_upload_task = asyncio.create_task(self._upload_crop_batch(crop_tasks, batch_counter))
            upload_tasks.append(final_upload_task)
        
        # Wait for any pending uploads to complete and collect results
        if upload_tasks:
            logger.info(f"Waiting for {len(upload_tasks)} pending upload tasks to complete before stopping...")
            
            async def wait_for_pending_uploads():
                await asyncio.gather(*upload_tasks)
            
            asyncio.run(wait_for_pending_uploads())

        # Return context with partial results
        context_to_update = {
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
        }
        return context_to_update
