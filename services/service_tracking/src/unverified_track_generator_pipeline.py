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
    from shared_libs.config.all_config import detection_config

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
import os
import cv2
import json
from matplotlib.style import context
import numpy as np
import asyncio
import time
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

# Suppress PyTorch JIT tracing warnings in multiprocessing context
warnings.filterwarnings('ignore', category=UserWarning, message='.*TracerWarning.*')

import supervision as sv
from supervision.utils.image import crop_image

from google.cloud import firestore
from shared_libs.common.google_storage import get_storage, GCSPaths
from shared_libs.common.detection import DetectionModel
from shared_libs.common.pipeline_step import StepStatus
from shared_libs.common.pipeline import Pipeline, PipelineStatus
from shared_libs.config.all_config import DetectionConfig, detection_config, model_config
from shared_libs.utils.id_generator import create_video_id, create_run_id
from shared_libs.common.tracker import AffineAwareByteTrack
from shared_libs.common.detection_utils import detections_to_json
from shared_libs.common.track_to_player import map_detections_to_players

logger = logging.getLogger(__name__)

#: Minimum video resolution required for processing (width, height)
MIN_VIDEO_RESOLUTION = (1920, 1080)
FRAME_SAMPLING_FOR_CROP = 15

#: Batch size for crop processing and upload (match process pool worker count)
CROP_BATCH_SIZE = 4

#: Maximum concurrent upload tasks to prevent overwhelming storage service
MAX_CONCURRENT_UPLOADS = 2

#: Interval for checkpoint saves (every N frames)
CHECKPOINT_FRAME_INTERVAL = 100

CROP_WORKER_ENV_VAR = "TRACKING_CROP_WORKERS"
CROP_BATCH_ENV_VAR = "TRACKING_CROP_BATCH_SIZE"


def _determine_crop_concurrency() -> Tuple[int, int]:
    """Derive process worker count and crop batch size from CPU info or overrides."""
    default_workers = CROP_BATCH_SIZE
    worker_count = default_workers

    env_worker_value = os.getenv(CROP_WORKER_ENV_VAR)
    if env_worker_value:
        try:
            worker_count = max(1, int(env_worker_value))
        except ValueError:
            logger.warning(
                "Invalid %s value '%s'; falling back to default %d workers",
                CROP_WORKER_ENV_VAR,
                env_worker_value,
                default_workers,
            )
            worker_count = default_workers
    else:
        cpu_total = os.cpu_count()
        if cpu_total and cpu_total > 0:
            if cpu_total <= 2:
                worker_count = max(1, cpu_total)
            else:
                worker_count = max(default_workers, cpu_total - 1)

    env_batch_value = os.getenv(CROP_BATCH_ENV_VAR)
    if env_batch_value:
        try:
            batch_size = max(1, int(env_batch_value))
        except ValueError:
            logger.warning(
                "Invalid %s value '%s'; using worker count %d",
                CROP_BATCH_ENV_VAR,
                env_batch_value,
                worker_count,
            )
            batch_size = worker_count
    else:
        batch_size = worker_count

    if batch_size > worker_count:
        logger.warning(
            "Crop batch size %d exceeds worker count %d; capping batch size",
            batch_size,
            worker_count,
        )
        batch_size = worker_count

    return worker_count, batch_size


def extract_crops_from_frame(
    shm_name: str,
    frame_shape: Tuple[int, int, int],
    frame_dtype: str,
    detections_dict: Union[Dict[str, Any], List[Dict[str, Any]]],
    frame_number: int,
    min_width: int,
    min_height: int,
    min_contrast: float
) -> Tuple[List[Tuple[int, bytes]], float]:
    """Extract quality-filtered crops from shared memory frame data.

    This function is defined at module scope so it can be pickled by
    :class:`concurrent.futures.ProcessPoolExecutor`. It attaches to the
    shared memory segment created by the caller, reconstructs the frame, and
    iterates through serialized detections to generate JPEG-encoded crops.

    Args:
        shm_name: Name of the shared memory block containing frame bytes.
        frame_shape: Shape tuple ``(height, width, channels)`` for the frame.
        frame_dtype: NumPy dtype string (e.g., ``"uint8"``) for the frame.
    detections_dict: Serialized detections payload (dict or list of dicts).
    frame_number: Frame identifier being processed (for instrumentation).
        min_width: Minimum crop width in pixels.
        min_height: Minimum crop height in pixels.
        min_contrast: Minimum grayscale standard deviation required.

    Returns:
        List of ``(track_id, jpeg_bytes)`` tuples for crops passing quality checks.
    """
    import logging
    import time
    from multiprocessing import shared_memory
    from shared_libs.common.detection_utils import json_to_detections

    worker_logger = logging.getLogger(__name__)
    results: List[Tuple[int, bytes]] = []
    shm = None

    try:
        # Attach to shared memory created by the parent process
        shm = shared_memory.SharedMemory(name=shm_name)
        dtype = np.dtype(frame_dtype)
        frame = np.ndarray(frame_shape, dtype=dtype, buffer=shm.buf)

        if isinstance(detections_dict, list):
            detections = json_to_detections(detections_dict)
        else:
            detections = json_to_detections([detections_dict])
        num_detections = len(detections)

        if num_detections == 0:
            worker_logger.info(
                "[WORKER %s] Frame %s has no detections; skipping",
                shm_name,
                frame_number,
            )
            return results, 0.0

        worker_logger.debug(
            "[WORKER %s] Processing %d detections via shared memory",
            shm_name,
            num_detections,
        )

        start_time = time.time()
        worker_logger.info(
            "[WORKER %s] Start crop extraction | frame=%s | detections=%d",
            shm_name,
            frame_number,
            num_detections,
        )

        for xyxy, _mask, confidence, class_id, tracker_id, data in detections:
            try:
                crop = crop_image(frame, xyxy.astype(int))
            except Exception as exc:  # pragma: no cover - defensive
                worker_logger.debug("Failed to crop detection: %s", exc)
                continue

            height, width = crop.shape[:2]
            if width < min_width or height < min_height:
                continue

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            contrast = float(np.asarray(gray_crop, dtype=np.float32).std())
            if contrast < min_contrast:
                continue

            success, buffer = cv2.imencode(".jpg", crop)
            if not success:
                worker_logger.debug("cv2.imencode failed for tracker_id=%s", tracker_id)
                continue

            track_id = int(tracker_id) if tracker_id is not None else -1
            results.append((track_id, buffer.tobytes()))
        processing_time = time.time() - start_time
        worker_logger.info(
            "[WORKER %s] Finished crop extraction | frame=%s | crops=%d | duration=%.3fs",
            shm_name,
            frame_number,
            len(results),
            processing_time,
        )

        return results, processing_time

    except Exception as exc:  # pragma: no cover - worker side logging
        worker_logger.error("Shared memory crop extraction failed: %s", exc)
        return results, 0.0

    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                worker_logger.debug("Failed to close shared memory %s", shm_name)


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
        from shared_libs.config.all_config import detection_config

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
    MIN_CROP_WIDTH = 10
    MIN_CROP_HEIGHT = 20
    MIN_CROP_CONTRAST = 20.0

    def __init__(self, 
                 config: DetectionConfig, 
                 tenant_id: str, 
                 verbose: bool = True, 
                 enable_grass_mask: bool = model_config.enable_grass_mask, 
                 delete_process_folder: bool = True, 
                 task_id: Optional[str] = None,
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
            enable_grass_mask (bool): Enable background removal functionality using statistical color analysis.
                Currently unused in this pipeline
            delete_process_folder (bool): Clean up temporary processing files after completion to save storage
            task_id (Optional[str]): Task ID for progress tracking and status updates
            **kwargs: Additional arguments passed to the parent Pipeline class
        
        Raises:
            RuntimeError: If the detection model or tracker fails to initialize
            ValueError: If tenant_id is empty or config contains invalid parameters
        """
        self.config = config
        self.tenant_id = tenant_id
        self.video_capture = None
        self.delete_process_folder = delete_process_folder
        self.dataloader_workers = 2
        
        # Import transform_config to get the background removal setting
        from shared_libs.config.all_config import transform_config
        
        # Get storage client
        self.tenant_storage = get_storage(tenant_id)  # For tenant-specific operations (without /user suffix)
        
        # Initialize GCS path manager for structured path handling
        self.path_manager = GCSPaths()
        
        # Store tenant_id for path generation
        self.tenant_id = tenant_id
        self.task_id = task_id
        
        # Initialize Firestore client for progress updates
        if task_id:
            try:
                self.firestore_client = firestore.Client()
                self.progress_collection = self.firestore_client.collection('tracking_progress')
                logger.info(f"Initialized Firestore progress tracking for task_id: {task_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Firestore client for progress tracking: {e}")
                self.firestore_client = None
        else:
            self.firestore_client = None
        
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
        
        # Initialize process pool executor for parallel crop extraction
        # Process pool bypasses GIL for true parallelism on CPU-bound crop operations
        self._crop_worker_count, self._crop_batch_size = _determine_crop_concurrency()
        self._crop_executor = ProcessPoolExecutor(
            max_workers=self._crop_worker_count
        )
        logger.info(
            "Crop extraction process pool initialized (workers=%d, batch_size=%d)",
            self._crop_worker_count,
            self._crop_batch_size,
        )

        # Semaphores (created lazily within the running event loop)
        self._upload_semaphore = None  # type: Optional[asyncio.Semaphore]
        self._crop_task_semaphore = None  # type: Optional[asyncio.Semaphore]
        self._active_crop_tasks: int = 0

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
            verbose=verbose
        )
        
        # Override run_folder to use structured GCS path
        self.run_guid = create_run_id()
        self.run_folder = self.path_manager.get_path("run_data", run_id=self.run_guid)
    
    def __del__(self):
        """Cleanup resources on pipeline destruction."""
        if hasattr(self, '_crop_executor'):
            self._crop_executor.shutdown(wait=False)
            logger.debug("Crop extraction process pool shut down")

    def _update_progress(self, progress_data: Dict[str, Any]) -> None:
        """Update progress in Firestore for real-time tracking."""
        if not self.firestore_client or not self.task_id:
            return
        
        try:
            doc_ref = self.progress_collection.document(self.task_id)
            progress_data['updated_at'] = datetime.now(timezone.utc).isoformat()
            doc_ref.set(progress_data, merge=True)
            logger.debug(f"Updated progress for task {self.task_id}: {progress_data}")
        except Exception as e:
            logger.warning(f"Failed to update progress in Firestore: {e}")

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

        logger.debug(f"Starting parallel {operation_func.__name__} of {len(tasks)} items{' for ' + context_info if context_info else ''}")
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

    def run(self, video_path: Optional[str], resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Execute the complete data preparation pipeline for a single video.
        
        When resuming from a checkpoint, the pipeline loads the video path/blob name from the checkpoint context, not from the new video_path argument. If a different video_path is supplied, it is ignored and a warning is logged.
        """

        # 1. If video_path is provided, use it
        if video_path:
            logger.info(f"Starting data preparation pipeline for video: {video_path}")
            context = {"raw_video_path": video_path}
            results = super().run(context, resume_from_checkpoint=resume_from_checkpoint)
            context = results.get("context", {})
            checkpoint_video_path = None
            if resume_from_checkpoint and context.get("resume_frame") is not None:
                if context.get("video_blob_name"):
                    checkpoint_video_path = context["video_blob_name"]
                elif context.get("raw_video_path"):
                    checkpoint_video_path = context["raw_video_path"]
                if checkpoint_video_path and checkpoint_video_path != video_path:
                    logger.warning(f"Resuming from checkpoint: supplied video_path '{video_path}' is ignored. Using checkpoint video path '{checkpoint_video_path}' instead.")
            resolved_video_path = checkpoint_video_path if checkpoint_video_path else video_path
        else:
            # 2. If not, use resume_from_checkpoint
            if not resume_from_checkpoint:
                logger.warning("No video_path provided and resume_from_checkpoint is False. Cannot proceed.")
                return {"status": PipelineStatus.ERROR.value, "error": "No video path provided and resume_from_checkpoint is False."}
            logger.info("No video_path provided. Attempting to resume from checkpoint...")
            context = {}
            results = super().run(context, resume_from_checkpoint=True)
            context = results.get("context", {})
            checkpoint_video_path = None
            if context.get("video_blob_name"):
                checkpoint_video_path = context["video_blob_name"]
            elif context.get("raw_video_path"):
                checkpoint_video_path = context["raw_video_path"]
            if not checkpoint_video_path:
                logger.warning("resume_from_checkpoint is True but no video found in checkpoint context. Cannot proceed.")
                return {"status": PipelineStatus.ERROR.value, "error": "resume_from_checkpoint is True but no video found in checkpoint context."}
            resolved_video_path = checkpoint_video_path

        formatted_results = {
            "status": results["status"],
            "run_guid": results["run_guid"],
            "run_folder": results["run_folder"],
            "video_path": resolved_video_path,
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

    async def _save_detections_async(self, detections_path: str, detections_data: sv.Detections) -> None:
        """
        Save detections asynchronously without blocking the main processing flow.
        
        Args:
            detections_path: GCS path where to save the detections
            detections_data: List of detections to save
        """
        try:
            json_data = detections_to_json(detections_data)
            json_bytes = json.dumps(json_data).encode('utf-8')
            self.storage_client.upload_from_bytes(detections_path, json_bytes)
            logger.debug(f"Asynchronously saved detections to {detections_path}")
        except Exception as e:
            logger.error(f"Failed to save detections asynchronously to {detections_path}: {e}")

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
        from shared_libs.common.pipeline import stop_pipeline
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
            video_guid = create_video_id(raw_video_path)
            
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
        
        # Run the async processing within a new event loop
        return asyncio.run(self._process_video_frames_async(context))

    def _log_batch_metrics(self, batch_num: int, batch_size: int, detection_time: float, 
                          tracking_time: float, total_detections: int, frame_start: int):
        """
        Log comprehensive batch processing metrics for GPU utilization analysis.
        
        Args:
            batch_num: Current batch number
            batch_size: Number of frames in this batch
            detection_time: Time spent on batch detection (seconds)
            tracking_time: Time spent on sequential tracking (seconds)
            total_detections: Total detections found in batch
            frame_start: Starting frame number of this batch
        """
        # Calculate metrics
        avg_detection_time_per_frame = detection_time / batch_size if batch_size > 0 else 0
        avg_tracking_time_per_frame = tracking_time / batch_size if batch_size > 0 else 0
        total_time = detection_time + tracking_time
        detection_fps = batch_size / detection_time if detection_time > 0 else 0
        tracking_fps = batch_size / tracking_time if tracking_time > 0 else 0
        total_fps = batch_size / total_time if total_time > 0 else 0
        
        # Log detailed metrics for cloud monitoring
        logger.info(
            f"📊 BATCH {batch_num} METRICS | "
            f"Frames: {frame_start}-{frame_start + batch_size - 1} ({batch_size} frames) | "
            f"Detections: {total_detections}"
        )
        logger.info(
            f"⚡ BATCH {batch_num} PERFORMANCE | "
            f"Detection: {detection_time:.3f}s ({detection_fps:.1f} fps) | "
            f"Tracking: {tracking_time:.3f}s ({tracking_fps:.1f} fps) | "
            f"Total: {total_time:.3f}s ({total_fps:.1f} fps)"
        )
        logger.info(
            f"🔢 BATCH {batch_num} PER-FRAME | "
            f"Detection: {avg_detection_time_per_frame*1000:.1f}ms/frame | "
            f"Tracking: {avg_tracking_time_per_frame*1000:.1f}ms/frame | "
            f"Avg detections/frame: {total_detections/batch_size if batch_size > 0 else 0:.1f}"
        )
    
    def _log_crop_metrics(
        self,
        crop_time: float,
        num_crops: int,
        frame_idx: int,
        worker_time: Optional[float] = None,
        wait_time: Optional[float] = None,
    ) -> None:
        """
        Log crop extraction performance metrics.
        
        Args:
            crop_time: Time spent extracting crops (seconds)
            num_crops: Number of crops extracted
            frame_idx: Frame number being processed
            worker_time: Actual processing time spent inside worker (seconds)
            wait_time: Time spent waiting in the executor queue (seconds)
        """
        crops_per_sec = num_crops / crop_time if crop_time > 0 else 0
        timing_detail = ""
        if worker_time is not None and wait_time is not None:
            timing_detail = (
                f" | worker {worker_time:.3f}s | wait {wait_time:.3f}s"
            )
        logger.info(
            f"✂️ CROP Frame {frame_idx} | "
            f"{num_crops} crops in {crop_time:.3f}s "
            f"({crops_per_sec:.1f} crops/sec){timing_detail}"
        )

    async def _process_video_frames_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async method to process video frames with proper event loop handling.
        Supports both batch detection (GPU-optimized) and sequential detection (fallback).
        """
        video_blob_name = context.get("video_blob_name")
        video_guid = context.get("video_guid")
        video_folder = context.get("video_folder")
        
        if not video_guid:
            logger.error("No video GUID for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No video GUID provided"}
        
        if not video_folder:
            logger.error("No video folder for player detection")
            return {"status": StepStatus.ERROR.value, "error": "No video folder provided for saving detections"}
        
        try:
            logger.info(f"Starting player detection for video: {video_guid}")

            detections_path = self.path_manager.get_path("detections_path", video_id=video_guid)
            if not detections_path:
                logger.error("Could not determine detections path. Aborting.")
                return {"status": StepStatus.ERROR.value, "error": "Could not determine detections path"}
            
            # Process all frames in the video
            detections_count = 0
            all_detections = sv.Detections.empty()
            get_crop_tasks = []  # Collect async crop processing tasks
            upload_tasks = []  # Track concurrent upload tasks
            batch_counter = 0  # Track batch numbers for logging
            all_crop_paths = []  # Track all successful crop upload paths
           
            last_checkpoint_frame = -1
            
            # Check for frame-level resume information in context
            resume_frame = context.get("resume_frame", 0)
            resume_detections_count = context.get("resume_detections_count", 0)
            resume_all_detections = context.get("resume_all_detections", sv.Detections.empty())
            resume_crop_paths = context.get("resume_crop_paths", [])
            
            if resume_frame > 0:
                logger.info(f"Resuming video processing from frame {resume_frame}")
                detections_count = resume_detections_count
                all_detections = resume_all_detections
                all_crop_paths = resume_crop_paths
            
            with self.tenant_storage.get_video_capture(video_blob_name or "") as cap:
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
                
                # Determine if we should use batch detection (GPU optimization)
                use_batch_detection = self.config.use_batch_detection and self.detection_model.batch_size > 1
                
                if use_batch_detection:
                    logger.info(f"🚀 BATCH DETECTION ENABLED | Batch size: {self.detection_model.batch_size} frames | GPU optimization active")
                else:
                    logger.info(f"📦 SEQUENTIAL DETECTION | Processing frame-by-frame (batch_size={self.detection_model.batch_size})")
                
                # Update status to "running" now that we're about to process frames
                # This complies with PipelineStatus.RUNNING enum value
                self._update_progress({
                    'status': 'running',
                    'progress_percent': 0.0,
                    'frames_processed': resume_frame,
                    'total_frames': total_frames,
                    'detections_count': detections_count,
                    'current_video': video_guid
                })
                logger.info(f"Status updated to 'running' - starting frame processing")
                
                # Frame buffer for batch processing
                frame_buffer: List[np.ndarray] = []
                frame_indices_buffer: List[int] = []
                batch_num = 0
                
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
                    if not ret or frame is None or not isinstance(frame, np.ndarray):
                        if not ret or frame is None:
                            logger.debug(f"Frame read failed or returned None at frame {frame_number}. End of video reached.")
                        else:
                            logger.warning(f"Frame at {frame_number} is not a valid numpy array. Type: {type(frame)}")
                        
                        # Process any remaining frames in buffer before stopping
                        if use_batch_detection and len(frame_buffer) > 0:
                            logger.info(f"Processing final batch of {len(frame_buffer)} frames")
                            # Process final partial batch using same logic as full batch
                            batch_detections_list = self.detection_model.generate_detections_batch(frame_buffer)  # type: ignore
                            for i, (frame_rgb_buffered, detections, frame_idx) in enumerate(
                                zip(frame_buffer, batch_detections_list, frame_indices_buffer)
                            ):
                                if detections is None or len(detections) == 0:
                                    detections = sv.Detections.empty()
                                else:
                                    detections_count += len(detections)
                                
                                if frame_idx == 0:
                                    affine_matrix = self.tracker.get_identity_affine_matrix()
                                else:
                                    affine_matrix = self.tracker.calculate_affine_transform(
                                        previous_frame_rgb, frame_rgb_buffered
                                    )
                                detections = self.tracker.update_with_transform(
                                    detections, affine_matrix, frame_rgb_buffered
                                )
                                previous_frame_rgb = frame_rgb_buffered
                                
                                if len(detections) > 0:
                                    detections.data['frame_index'] = [frame_idx] * len(detections)
                                else:
                                    detections.data['frame_index'] = []
                                all_detections = sv.Detections.merge([all_detections, detections])
                        break

                    try:
                        # Initialize detections to empty to avoid unbound variable errors
                        detections = sv.Detections.empty()
                        
                        # Convert BGR to RGB for model input
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #type: ignore
                        
                        if use_batch_detection:
                            # === BATCH DETECTION MODE ===
                            # Buffer frames until we reach batch size, then process all at once
                            frame_buffer.append(frame_rgb)
                            frame_indices_buffer.append(frame_number)
                            
                            # Process batch when buffer is full
                            if len(frame_buffer) >= self.detection_model.batch_size:
                                batch_start_time = time.time()
                                
                                # Step 1: Batch Detection (GPU-accelerated)
                                detection_start = time.time()
                                batch_detections_list = self.detection_model.generate_detections_batch(frame_buffer)  # type: ignore
                                detection_time = time.time() - detection_start
                                
                                # Step 2: Sequential Tracking (maintains temporal consistency)
                                tracking_start = time.time()
                                batch_detections_tracked = []
                                batch_total_detections = 0
                                
                                for i, (frame_rgb_buffered, detections, frame_idx) in enumerate(
                                    zip(frame_buffer, batch_detections_list, frame_indices_buffer)
                                ):
                                    # Validate detections
                                    if type(detections) is not sv.Detections:
                                        raise RuntimeError("Unrecognized detection type")
                                    if detections is None or len(detections) == 0:
                                        logger.debug(f"Frame {frame_idx}: No detections found")
                                        detections = sv.Detections.empty()
                                    else:
                                        batch_total_detections += len(detections)
                                        logger.debug(f"Frame {frame_idx}: Found {len(detections)} detections")

                                    # Calculate affine transformation and update tracker
                                    if frame_idx == 0:
                                        affine_matrix = self.tracker.get_identity_affine_matrix()
                                    else:
                                        affine_matrix = self.tracker.calculate_affine_transform(
                                            previous_frame_rgb, frame_rgb_buffered
                                        )

                                    detections = self.tracker.update_with_transform(
                                        detections, affine_matrix, frame_rgb_buffered
                                    )
                                    batch_detections_tracked.append((frame_rgb_buffered, detections, frame_idx))
                                    previous_frame_rgb = frame_rgb_buffered
                                
                                tracking_time = time.time() - tracking_start
                                detections_count += batch_total_detections
                                
                                # Log batch metrics
                                self._log_batch_metrics(
                                    batch_num, len(frame_buffer), detection_time, tracking_time,
                                    batch_total_detections, frame_indices_buffer[0]
                                )
                                
                                # Step 3: Handle crops and save detections for each frame in batch
                                for frame_rgb_tracked, detections_tracked, frame_idx in batch_detections_tracked:
                                    if frame_idx % FRAME_SAMPLING_FOR_CROP == 0:
                                        get_crop_task = asyncio.create_task(
                                            self._get_upload_list_for_frame(
                                                frame_rgb_tracked, detections_tracked, video_guid, frame_idx
                                            )
                                        )
                                        get_crop_tasks.append(get_crop_task)
                                        
                                        if len(get_crop_tasks) >= self._crop_batch_size:
                                            logger.debug(f"Processing crop batch {batch_counter} (frame {frame_idx})...")
                                            batch_frame_tasks = get_crop_tasks[:]
                                            get_crop_tasks.clear()
                                            upload_task = asyncio.create_task(
                                                self._upload_crop_batch(batch_frame_tasks, batch_counter)
                                            )
                                            upload_tasks.append(upload_task)
                                            batch_counter += 1
                                            await asyncio.sleep(0)
                                    
                                    # Set frame_index and merge detections
                                    if len(detections_tracked) > 0:
                                        detections_tracked.data['frame_index'] = [frame_idx] * len(detections_tracked)
                                    else:
                                        detections_tracked.data['frame_index'] = []
                                    all_detections = sv.Detections.merge([all_detections, detections_tracked])
                                
                                # Save detections asynchronously
                                asyncio.create_task(self._save_detections_async(detections_path, all_detections))
                                
                                # Clear buffers for next batch
                                frame_buffer.clear()
                                frame_indices_buffer.clear()
                                batch_num += 1
                                frame_number += 1
                                
                                # Checkpointing
                                if frame_number % CHECKPOINT_FRAME_INTERVAL == 0 and frame_number != last_checkpoint_frame:
                                    checkpoint_context = {
                                        "resume_frame": frame_number,
                                        "resume_detections_count": detections_count,
                                        "resume_all_detections": all_detections,
                                        "resume_crop_paths": all_crop_paths,
                                    }
                                    if not self.save_checkpoint(checkpoint_context, self.current_completed_steps):
                                        logger.warning(f"Failed to save checkpoint at frame {frame_number}")
                                    last_checkpoint_frame = frame_number
                                
                                # Progress logging
                                if frame_number % 100 == 0:
                                    logger.info(f"📹 Progress: {frame_number}/{total_frames} frames ({detections_count} detections)")
                                    self._update_progress({
                                        'status': 'running',  # Use PipelineStatus.RUNNING.value
                                        'progress_percent': (frame_number / total_frames) * 100,
                                        'frames_processed': frame_number,
                                        'total_frames': total_frames,
                                        'detections_count': detections_count,
                                        'current_video': video_guid
                                    })
                                continue  # Skip the sequential processing below
                            else:
                                # Buffer not full yet, continue reading frames
                                frame_number += 1
                                continue
                        
                        # === SEQUENTIAL DETECTION MODE (fallback) ===
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

                        detections = self.tracker.update_with_transform(detections, affine_matrix, frame_rgb)    
                        if frame_number % FRAME_SAMPLING_FOR_CROP == 0:
                            # Create async task for crop processing (returns list of (blob_path, crop_image))
                            get_crop_task = asyncio.create_task(
                                self._get_upload_list_for_frame(frame_rgb, detections, video_guid, frame_number)
                            )
                            get_crop_tasks.append(get_crop_task)
                            logger.debug(f"Crop task created (pending={len(get_crop_tasks)} in current batch) -> {get_crop_task}")

                            if len(get_crop_tasks) >= self._crop_batch_size:
                                logger.info(f"Processing batch {batch_counter} of crop uploads (frame {frame_number})...")
                                # Snapshot current tasks and reset collector immediately to avoid mutation during gather
                                batch_frame_tasks = get_crop_tasks[:]
                                get_crop_tasks.clear()
                                upload_task = asyncio.create_task(
                                    self._upload_crop_batch(batch_frame_tasks, batch_counter)
                                )
                                logger.debug(f"Upload batch task {upload_task} created for batch {batch_counter}")
                                upload_tasks.append(upload_task)
                                batch_counter += 1
                                await asyncio.sleep(0)

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
                        all_detections = sv.Detections.merge([all_detections, detections])
                        # Save detections asynchronously (non-blocking)
                        asyncio.create_task(self._save_detections_async(detections_path, all_detections))

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
                        # Update progress in Firestore for real-time tracking
                        progress_percent = (frame_number / total_frames) * 100
                        self._update_progress({
                            'status': 'running',  # Use PipelineStatus.RUNNING.value
                            'progress_percent': progress_percent,
                            'frames_processed': frame_number,
                            'total_frames': total_frames,
                            'detections_count': detections_count,
                            'current_video': video_guid if 'video_guid' in locals() else None
                        })
            
            # Process any remaining crops in final batch
            if len(get_crop_tasks) > 0:
                final_upload_task = asyncio.create_task(self._upload_crop_batch(get_crop_tasks, batch_counter))
                upload_tasks.append(final_upload_task)
                batch_counter += 1
            
            # Wait for all upload tasks to complete and collect results
            if upload_tasks:
                logger.info(f"Waiting for {len(upload_tasks)} upload batches to complete...")
                
                # Wait for all upload batch tasks to complete
                await asyncio.gather(*upload_tasks)
                
                logger.info(f"All crop upload batches completed")
            
            # Save final detections asynchronously (non-blocking)
            asyncio.run(self._save_detections_async(detections_path, all_detections))

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

    async def _upload_crop_batch(self, get_crop_tasks: List[asyncio.Task], batch_counter: int) -> List[str]:
        """
        Await frame-level crop tasks, flatten their results, and upload all crops.

        Args:
            get_crop_tasks: List of asyncio.Tasks, each resolving to List[(blob_path, crop_image)]
            batch_counter: Batch index for logging

        Returns:
            List[str]: Successfully uploaded blob paths
        """
        uploaded_paths: List[str] = []
        try:
            logger.debug(f"[UPLOAD_BATCH] Entered batch {batch_counter} with {len(get_crop_tasks)} frame tasks")
            # Wait for all frame crop extraction tasks
            crop_results = await asyncio.gather(*get_crop_tasks)
            logger.debug(f"[UPLOAD_BATCH] Batch {batch_counter} frame task gather complete")

            # Flatten results (skip empty lists)
            flat_crops = [item for sublist in crop_results if sublist for item in sublist]
            logger.debug(f"[UPLOAD_BATCH] Batch {batch_counter} flattened to {len(flat_crops)} crops")

            if not flat_crops:
                logger.debug(f"[UPLOAD_BATCH] Batch {batch_counter} has no crops to upload")
                return uploaded_paths

            # Prepare tasks for parallel upload
            failed_uploads, successful_uploads = await self._execute_parallel_operations_async(
                flat_crops,
                lambda blob_path, crop_img: self.tenant_storage.upload_from_bytes(
                    blob_path,
                    crop_img
                ),
                f"upload batch {batch_counter}"
            )

            # Derive successful paths
            if successful_uploads:
                failed_set = set(failed_uploads)
                for item in flat_crops:
                    if item not in failed_set:
                        uploaded_paths.append(item[0])

            logger.debug(f"[UPLOAD_BATCH] Batch {batch_counter}: {len(uploaded_paths)} uploaded, {len(failed_uploads)} failed")

            return uploaded_paths

        except Exception as e:
            logger.debug(f"[UPLOAD_BATCH] Exception in batch {batch_counter}: {e}")
            logger.debug(e)
            return uploaded_paths

    def _get_crops(self, frame: np.ndarray, detections: sv.Detections) -> List[Tuple[int, np.ndarray]]:
        """
        Extract quality-filtered crops from frame detections.
        
        Returns list of (track_id, crop_image) tuples for valid detections.
        """
        res = []
        try:
            num_detections = len(detections)
        except Exception:
            logger.error("Detections object is not iterable or invalid type.")
            return []
        if num_detections == 0:
            logger.warning("No detections found for frame; skipping crop extraction.")
        else:
            logger.debug(f"Extracting crops from {num_detections} detections")
        for detection in detections:
            crop = crop_image(frame, detection[0].astype(int))
            if self._is_crop_quality_sufficient(crop):
                crop_tuple = (detection[4], crop)  # (track_id, crop)
                res.append(crop_tuple)
        num_quality_crops = len(res)
        if num_detections > 0:
            discard_rate = 1 - (num_quality_crops / num_detections)
            if discard_rate > 0.2:
                logger.warning(f"Discarded {discard_rate*100:.1f}% of crops due to quality ({num_detections-num_quality_crops}/{num_detections})")
            logger.debug(f"Extracted {num_quality_crops} quality crops from {num_detections} detections")
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
            logger.info(f"Crop too small: {width}x{height} (min: {self.MIN_CROP_WIDTH}x{self.MIN_CROP_HEIGHT})")
            return False
        
        # Check contrast using standard deviation of grayscale image
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_crop)  #type: ignore
        
        # Low contrast threshold
        if contrast < self.MIN_CROP_CONTRAST:
            logger.info(f"Crop has low contrast: {contrast:.2f} < {self.MIN_CROP_CONTRAST}")
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
        frame_number: int) -> List[Tuple[str, bytes]]:
        """Generate upload tasks for crops extracted from a single frame."""
        from multiprocessing import shared_memory
        from shared_libs.common.detection_utils import detections_to_json

        if self._crop_task_semaphore is None:
            self._crop_task_semaphore = asyncio.Semaphore(self._crop_worker_count)

        async with self._crop_task_semaphore:
            self._active_crop_tasks += 1
            submit_time = time.time()
            logger.info(
                "✂️ SUBMIT Frame %s | detections=%d | active_workers=%d",
                frame_number,
                len(detections),
                self._active_crop_tasks,
            )

            shm = None
            worker_time: float = 0.0
            try:
                detections_dict = detections_to_json(detections)

                frame_bytes = frame.nbytes
                shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
                shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                shm_array[:] = frame

                loop = asyncio.get_event_loop()
                crop_start = time.time()

                logger.debug(
                    "Submitting crop extraction for frame %s via shared memory %s",
                    frame_number,
                    shm.name,
                )

                crop_tuples, worker_time = await loop.run_in_executor(
                    self._crop_executor,
                    extract_crops_from_frame,
                    shm.name,
                    frame.shape,
                    str(frame.dtype),
                    detections_dict,
                    frame_number,
                    self.MIN_CROP_WIDTH,
                    self.MIN_CROP_HEIGHT,
                    self.MIN_CROP_CONTRAST,
                )

                crop_time = time.time() - crop_start
                wait_time = max(crop_time - worker_time, 0.0)

                self._log_crop_metrics(
                    crop_time,
                    len(crop_tuples),
                    frame_number,
                    worker_time,
                    wait_time,
                )

                upload_list: List[Tuple[str, bytes]] = []
                if crop_tuples:
                    for track_id, jpeg_bytes in crop_tuples:
                        crop_dir = self.path_manager.get_path(
                            "unverified_tracks",
                            video_id=video_guid,
                            track_id=track_id,
                        )
                        if crop_dir is None:
                            raise RuntimeError(
                                f"Unable to determine crop directory for {video_guid}, track {track_id}"
                            )
                        crop_path = f"{crop_dir.rstrip('/')}/crop_{frame_number}.jpg"
                        upload_list.append((crop_path, jpeg_bytes))

                    logger.debug(
                        "Generated upload list for frame %s: %d crops",
                        frame_number,
                        len(upload_list),
                    )
                else:
                    logger.debug("No valid crops found for frame %s", frame_number)

                logger.info(
                    "✂️ COMPLETE Frame %s | total=%.3fs | worker=%.3fs | wait=%.3fs | crops=%d",
                    frame_number,
                    crop_time,
                    worker_time,
                    wait_time,
                    len(crop_tuples),
                )

                return upload_list

            except Exception as exc:
                logger.error(
                    "Error preparing crops for frame %s (video %s): %s",
                    frame_number,
                    video_guid,
                    exc,
                )
                logger.exception(exc)
                return []

            finally:
                if shm is not None:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception as cleanup_exc:
                        logger.warning(
                            "Shared memory cleanup failed for frame %s (%s): %s",
                            frame_number,
                            shm.name,
                            cleanup_exc,
                        )
                self._active_crop_tasks = max(self._active_crop_tasks - 1, 0)
                logger.info(
                    "✂️ RELEASE Frame %s | active_workers=%d | elapsed=%.3fs",
                    frame_number,
                    self._active_crop_tasks,
                    time.time() - submit_time,
                )

    def _graceful_stop(self, crop_tasks, upload_tasks, batch_counter, video_guid, 
                      all_crop_paths, frame_number, detections_count, all_detections,
                      video_blob_name, video_folder) -> Dict[str, Any]:
        """
        Handle graceful pipeline shutdown during video processing.

        Completes all pending crop extraction and upload tasks before stopping,
        preventing data loss and enabling resume capability. Updates Firestore status
        to 'cancelled' before shutting down.

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
            Updates Firestore status to 'cancelled' for external monitoring.
        """
        logger.info("Initiating graceful stop of the pipeline...")
        
        # Update Firestore status to cancelled immediately
        self._update_progress({
            'status': 'cancelled',
            'frames_processed': frame_number,
            'detections_count': detections_count,
            'cancellation_reason': 'Stop requested during detection processing'
        })
        logger.info(f"Updated Firestore status to 'cancelled' for task_id: {self.task_id}")
        
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

        logger.info(f"Graceful stop completed. Processed {frame_number} frames with {detections_count} detections.")

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
