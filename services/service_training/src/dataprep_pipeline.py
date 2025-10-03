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
    from shared_libs.config.all_config import detection_config

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

import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import cv2
import supervision as sv
from shared_libs.config.all_config import (
    DetectionConfig,
    detection_config,
    model_config,
    training_config,
)
from PIL import Image
from supervision import Detections
from shared_libs.common.detection_utils import detections_to_json
from supervision.utils.image import crop_image

from shared_libs.common.background_mask import (
    BackgroundMaskDetector, create_frame_generator_from_images)
from shared_libs.common.detection import DetectionModel
from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.common.pipeline import Pipeline, PipelineStatus
from shared_libs.common.pipeline_step import StepStatus
from shared_libs.utils.id_generator import (create_aug_crop_id, create_crop_id,
                                            create_dataset_id, create_frame_id,
                                            create_run_id, create_video_id)

from .augmentation import augment_images

logger = logging.getLogger(__name__)

#: Minimum video resolution required for processing (width, height)
MIN_VIDEO_RESOLUTION = (1920, 1080)


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
        from shared_libs.config.all_config import detection_config
        
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
                 enable_grass_mask: Optional[bool] = None, 
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
        self.frames_per_video = config.frames_per_video
        self.video_capture = None
        self.train_ratio = training_config.train_ratio
        self.delete_process_folder = delete_process_folder
        self.dataloader_workers = training_config.num_workers
        
        # Import transform_config to get the background removal setting
        from .config.all_config import transform_config

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
            logger.info("Detection model loaded successfully")
        except RuntimeError as e:
            logger.critical(
                f"CRITICAL ERROR: Detection model is required for training pipeline "
                f"but failed to load: {e}"
            )
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
            pipeline_name="dataprep_pipeline",
            storage_client=self.tenant_storage,
            step_definitions=step_definitions,
            verbose=verbose
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

        logger.info(
            f"Starting parallel {operation_func.__name__} of {len(tasks)} items"
            f"{' for ' + context_info if context_info else ''}"
        )
        failed_operations = []
        
        with ThreadPoolExecutor(max_workers=self.dataloader_workers) as executor:
            # Submit tasks using the provided function
            # For upload: operation_func(*task) - blob_name, data  
            # For move: operation_func(*task) - source, destination
            # For delete: operation_func(task) - blob_name (string)
            future_to_task = {
                executor.submit(operation_func, *task) if isinstance(task, tuple) 
                else executor.submit(operation_func, task): task
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
            logger.warning(
                f"Failed to {operation_func.__name__} {len(failed_operations)} out of "
                f"{len(tasks)} items{' for ' + context_info if context_info else ''}"
            )

        logger.info(
            f"Completed {successful_count} {operation_func.__name__} operations"
            f"{' for ' + context_info if context_info else ''}"
        )

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
            video_guid = create_video_id(raw_video_path)
            
            video_folder = self.path_manager.get_path("imported_video", video_id=video_guid)
            if video_folder is None:
                raise RuntimeError(f"Failed to generate video folder path for video_id: {video_guid}")
            video_folder = video_folder.rstrip('/')
            
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
    
    
    def _detect_players(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run player detection on the extracted frames using YOLO-based model.
        
        This method processes each extracted frame through the detection model to identify
        players. It handles batch processing, confidence filtering, and saves detection
        results to structured GCS paths for later crop extraction.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - frames_data: List of extracted frame images as numpy arrays
                - frame_ids: List of frame identifiers corresponding to frames_data
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
                        

            # Save detections using shared utility
            detections_blob_name = f"{video_folder.rstrip('/')}/detections.json"
            json_data = [
                detections_to_json(detections)
                for detections in all_detections
            ]
            json_bytes = json.dumps(json_data).encode("utf-8")
            self.tenant_storage.upload_from_bytes(detections_blob_name, json_bytes)

            logger.info(f"Player detection completed for frame id {frame_id} - {detections_count} detections found")

            if detections_count == 0:
                logger.warning(f"No detections found for video {video_guid} - skipping crop extraction")
                context.update({
                    "status": StepStatus.COMPLETED.value,
                    "all_detections": all_detections
                })
                return context

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
        Extract and save player crops from detections to GCS storage.
        
        This method processes detection results to extract individual player crops
        from frames. It uses parallel processing to efficiently upload crops to
        structured GCS paths for later processing stages.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - all_detections: List of detection results for each frame
                - frames_data: List of extracted frame images as numpy arrays
                - video_guid: Unique identifier for the video being processed
                - frame_ids: List of frame identifiers
                - frames_guids: List of frame GUIDs for path generation
                
        Returns:
            Dict[str, Any]: Updated context with crop extraction results:
                - status: Step completion status
                - crops_by_frame: List of crop data organized by frame
                - total_crops: Total number of crops extracted
                - upload_tasks: List of GCS upload operations performed
                - error: Error message if extraction failed
                
        Note:
            Crops are saved with structured naming conventions and uploaded in
            parallel for efficiency. Failed uploads are logged but don't stop
            the pipeline.
        """

        all_detections = context.get("all_detections")
        frames_data = context.get("frames_data")
        video_guid = context.get("video_guid") 
        frame_ids = context.get("frame_ids") or []
        frames_guids = context.get("frames_guids") or []

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
                    
                    if blob_path is None:
                        logger.error(f"Failed to generate blob path for video {video_guid}, frame {frame_guid}")
                        raise RuntimeError(f"Failed to generate blob path for video {video_guid}, frame {frame_guid}")
                    
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
                logger.warning(
                    f"Failed to upload {len(failed_uploads)} out of {len(upload_tasks)} "
                    f"crops for {video_guid}"
                )
            
            logger.info(f"Successfully uploaded {crops_uploaded} crops across {len(frames_data)} frames")
            
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
        
        This method applies statistical color analysis to identify and remove grass/field
        backgrounds from player crops. The background removal is performed in-memory
        and modifies the crop data for subsequent processing stages.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - crops_by_frame: List of crop data organized by frame
                - video_guid: Unique identifier for the video being processed
                - grass_mask_initialized: Boolean indicating if detector is ready
                
        Returns:
            Dict[str, Any]: Updated context with background removal results:
                - status: Step completion status
                - crops_wo_background: Number of crops processed
                - error: Error message if background removal failed
                
        Note:
            This step is conditionally executed based on enable_grass_mask configuration.
            Background removal happens in-memory and doesn't create new files in storage.
        """
       
        if not self.enable_grass_mask:
            logger.info("Grass mask disabled - skipping background removal step")
            return {"status": StepStatus.ERROR.value, "error": "Background removal step called but grass mask is disabled"}
        
        crops_by_frame = context.get("crops_by_frame")
        video_guid = context.get("video_guid")

        if not crops_by_frame:
            logger.error("No crops by frame data found for background removal")
            return {"status": StepStatus.ERROR.value, "error": "No crops by frame data provided"}
        
        if self.background_mask_detector is None:
            logger.error("Background mask detector not initialized")
            return {"status": StepStatus.ERROR.value, "error": "Background mask detector not initialized"}
        
        crops_processed = 0

        try:
            logger.info(f"Removing background from crops for video: {video_guid}")
            for frame in crops_by_frame:
                for crop in frame:
                      processed_crop = self.background_mask_detector.remove_background(
                                crop, 
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
        
        This method applies configured augmentation transforms to each original crop
        to increase dataset diversity. Augmented crops are saved with structured naming
        and uploaded in parallel for efficiency.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - video_guid: Unique identifier for the video being processed
                - frames_guids: List of frame GUIDs for path generation
                - crops_by_frame: List of crop data organized by frame
                
        Returns:
            Dict[str, Any]: Updated context with augmentation results:
                - status: Step completion status
                - crops_augmented: Boolean indicating if augmentation was performed
                - n_augmented_crops: Total number of augmented crops created
                - error: Error message if augmentation failed
                
        Note:
            Each original crop generates multiple augmented versions based on the
            configured augmentation pipeline. All augmented crops are uploaded
            in parallel for efficiency.
        """
        video_guid = context.get("video_guid")
        frames_guid = context.get("frames_guids")
        crops_by_frame = context.get("crops_by_frame")

        if not crops_by_frame or not frames_guid:
            logger.error("No crops by frame or frames GUID data found for augmentation")
            return {"status": StepStatus.ERROR.value, "error": "No crops by frame or frames GUID data provided"}

        
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
                    if blob_path is None:
                        logger.error(f"Failed to generate blob path for augmented crops: video {video_guid}, frame {frames_guid}, crop {temp_crop_guid}")
                        raise RuntimeError(f"Failed to generate blob path for augmented crops: video {video_guid}, frame {frames_guid}, crop {temp_crop_guid}")
                    
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
        
        This method organizes augmented crops into structured training and validation
        datasets. It creates separate datasets for each frame, splits players randomly
        based on the configured train_ratio, and moves crops to appropriate folders.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - crops_augmented: Boolean indicating if crops were augmented
                - frames_guids: List of frame GUIDs for processing
                - video_guid: Unique identifier for the video being processed
                
        Returns:
            Dict[str, Any]: Updated context with dataset creation results:
                - status: Step completion status
                - datasets_created: List of created dataset information
                - total_train_samples: Total number of training samples across all datasets
                - total_val_samples: Total number of validation samples across all datasets
                - error: Error message if dataset creation failed
                
        Note:
            Each frame generates its own dataset with independent train/val splits.
            The train_ratio configuration determines the split proportions.
        """
        crops_augmented = context.get("crops_augmented", False)
        frames_guids = context.get("frames_guids")
        video_guid = context.get("video_guid")
        
        if not crops_augmented:
            logger.warning("No augmented crops to create datasets from")
            return {"status": StepStatus.ERROR.value, "error": "No augmented crops available"}
        
        if not frames_guids:
            logger.error("No frames GUIDs found for dataset creation")
            return {"status": StepStatus.ERROR.value, "error": "No frames GUIDs provided"}
        
        try:
            logger.info(f"Creating training and validation sets for video {video_guid}")

            for frame_guid in frames_guids:

                aug_crops_root = self.path_manager.get_path("augmented_crops_root", video_id=video_guid, frame_id=frame_guid)
                if aug_crops_root is None:
                    logger.error(f"No augmented crops root found for video {video_guid} at frame {frame_guid}")
                    raise RuntimeError(f"No augmented crops root found for video {video_guid} at frame {frame_guid}")
                aug_crops_root = aug_crops_root.rstrip('/')
                players = self.tenant_storage.list_blobs(prefix=aug_crops_root, delimiter='/', exclude_prefix_in_return=True)
                players = list(players)  
                random.shuffle(players)  

                dataset_guid = create_dataset_id()
                n_train_players = int(len(players) * self.train_ratio)
                train_players = players[:n_train_players]
                val_players = players[n_train_players:]

                train_folder = self.path_manager.get_path("train_dataset", dataset_id=dataset_guid)
                if train_folder is None:
                    logger.error(f"Failed to generate train folder path for dataset {dataset_guid}")
                    raise RuntimeError(f"Failed to generate train folder path for dataset {dataset_guid}")
                train_folder = train_folder.rstrip('/')
                
                val_folder = self.path_manager.get_path("val_dataset", dataset_id=dataset_guid)
                if val_folder is None:
                    logger.error(f"Failed to generate val folder path for dataset {dataset_guid}")
                    raise RuntimeError(f"Failed to generate val folder path for dataset {dataset_guid}")
                val_folder = val_folder.rstrip('/')


                move_tasks = []
                for player in train_players:
                    player = player.rstrip('/')  # Ensure no trailing slash
                    player_aug_folder = self.path_manager.get_path("augmented_crops", video_id=video_guid, frame_id=frame_guid, orig_crop_id=player)
                    player_images_path_list = self.tenant_storage.list_blobs(prefix=player_aug_folder)
                    destination_folder = f"{train_folder}/{player}"
                  
                    for image_path in player_images_path_list:
                        if not image_path.endswith('.jpg'):
                            logger.warning(f"Skipping non-JPG file: {image_path}")
                            continue
                        file_name = os.path.basename(image_path)
                        move_tasks.append((image_path, f"{destination_folder}/{file_name}"))

                for player in val_players:
                    player = player.rstrip('/')
                    player_aug_folder = self.path_manager.get_path("augmented_crops", video_id=video_guid, frame_id=frame_guid, orig_crop_id=player)
                    player_images_path_list = self.tenant_storage.list_blobs(prefix=player_aug_folder)
                    destination_folder = f"{val_folder}/{player}"

                    for image_path in player_images_path_list:
                        if not image_path.endswith('.jpg'):
                            logger.warning(f"Skipping non-JPG file: {image_path}")
                            continue
                        file_name = os.path.basename(image_path)
                        move_tasks.append((image_path, f"{destination_folder}/{file_name}"))

                failed_moves, successful_moves = self._execute_parallel_operations(
                    move_tasks, 
                    self.tenant_storage.move_blob,
                    f"crops to train/val datasets for {frame_guid}"
                )

                # Count successful moves by type
                ttl_train = len(self.tenant_storage.list_blobs(prefix=train_folder))
                ttl_val = len(self.tenant_storage.list_blobs(prefix=val_folder))

                if ttl_train == 0:
                    logger.warning(f"No training crops created for {frame_guid} - check if there are any crops in the source folders")
                if ttl_val == 0:
                    logger.warning(f"No validation crops created for {frame_guid} - check if there are any crops in the source folders")

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


    def _extract_video_metadata(self, cap, video_path: str = "") -> Dict[str, Any]:
        """
        Extract metadata from video file.
        
        Args:
            cap: OpenCV VideoCapture object
            video_path: Path to the video file for error messages
            
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
        
        This method extracts a configured number of frames from the video at equally
        spaced intervals. Frames are saved to structured GCS paths and prepared for
        the detection pipeline. Video resolution validation is performed before extraction.
        
        Args:
            context (Dict[str, Any]): Pipeline context containing:
                - video_blob_name: GCS path to the imported video file
                - video_guid: Unique identifier for the video being processed
                
        Returns:
            Dict[str, Any]: Updated context with frame extraction results:
                - status: Step completion status
                - frames_data: List of extracted frame images as numpy arrays
                - video_guid: Unique identifier for the video
                - frames_guids: List of unique identifiers for each extracted frame
                - frame_ids: List of frame positions in the original video
                - error: Error message if extraction failed
                
        Note:
            Frames are extracted at equally spaced intervals based on frames_per_video
            configuration. All frames are uploaded to GCS with structured naming.
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

            video_metadata = self._extract_video_metadata(cap, video_blob_name)

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
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame at position {frame_position}")
                    continue
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
                
                if frame_folder is None:
                    logger.error(f"Failed to generate frame folder path for video {video_guid}, frame {frame_guid}")
                    raise RuntimeError(f"Failed to generate frame folder path for video {video_guid}, frame {frame_guid}")
                
                # Save frame to Google Storage using structured path
                frame_filename = f"{frame_guid}.jpg"
                frame_blob_path = f"{frame_folder.rstrip('/')}/{frame_filename}"

                # Encode frame to bytes for upload
                _, encoded_frame = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_bytes = encoded_frame.tobytes()

                success = self.tenant_storage.upload_from_bytes(frame_blob_path, frame_bytes)

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
        
        # Additional safety check to ensure the detector is not None
        if self.background_mask_detector is None:
            logger.error("Background mask detector is None despite enable_grass_mask being True")
            return {"status": StepStatus.ERROR.value, "error": "Background mask detector not initialized"}
        
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


