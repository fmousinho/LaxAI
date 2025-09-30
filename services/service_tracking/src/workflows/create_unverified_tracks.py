"""
Unverified Track Generation Workflow for LaxAI.

This module contains the core workflow logic for generating unverified tracks
from lacrosse videos. It can be used by CLI, API, or other interfaces.
"""

import logging
import os
import signal
import sys
import threading
from typing import Any, Dict, List, Optional

# Ensure src directory is in path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

from shared_libs.common.google_storage import GCSPaths, get_storage  # noqa: E402
from unverified_track_generator_pipeline import TrackGeneratorPipeline  # noqa: E402
from shared_libs.config.all_config import DetectionConfig, detection_config  # noqa: E402


class UnverifiedTrackGenerationWorkflow:
    """
    Orchestrates the end-to-end unverified track generation process for LaxAI.

    This class handles the complete workflow from video discovery
    to track generation execution.
    """

    def __init__(self,
                 tenant_id: str,
                 verbose: bool = True,
                 save_intermediate: bool = False,
                 custom_name: str = "track_generation_workflow_run",
                 resume_from_checkpoint: bool = True,
                 detection_config_param: Optional[DetectionConfig] = None,
                 task_id: Optional[str] = None,
                 cancellation_event: Optional[threading.Event] = None,
                 video_limit: Optional[int] = None):
        """
        Initialize the track generation workflow.

        Args:
            tenant_id: The tenant ID for GCS operations.
            verbose: Enable verbose logging for pipelines.
            save_intermediate: Save intermediate pipeline results to GCS.
            custom_name: Custom name for the track generation run.
            resume_from_checkpoint: Resume track generation from checkpoint if available.
            detection_config_param: Detection configuration object. If None, uses default.
            task_id: Task ID for tracking this track generation run.
            cancellation_event: Event for graceful cancellation handling.
            video_limit: Maximum number of videos to process (None for all).
        """
        self.tenant_id = tenant_id
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.custom_name = custom_name
        self.resume_from_checkpoint = resume_from_checkpoint
        self.detection_config = detection_config_param or detection_config
        self.task_id = task_id
        self.cancellation_event = cancellation_event
        self.video_limit = video_limit

        # Initialize Firestore client for status updates if task_id is provided
        self.firestore_client = None
        if self.task_id:
            try:
                from google.cloud import firestore
                self.firestore_client = firestore.Client()
                logger.info(f"Initialized Firestore client for task_id: {self.task_id}")
            except ImportError:
                logger.warning("google-cloud-firestore not available, status updates disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Firestore client: {e}")

        # Set up signal handlers for external cancellation (e.g., Cloud Run job cancellation)
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful cancellation."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            # Update Firestore status to cancelled if we have a task_id
            if self.task_id and self.firestore_client:
                try:
                    self._update_firestore_status("cancelled", f"Cancelled by signal {signum}")
                    logger.info(f"Updated Firestore status to cancelled for task_id: {self.task_id}")
                except Exception as e:
                    logger.error(f"Failed to update Firestore status on signal: {e}")

            # Set the cancellation event if it exists
            if self.cancellation_event:
                self.cancellation_event.set()
                logger.info("Set cancellation event")

            # Stop any active pipelines
            try:
                from shared_libs.common.pipeline import stop_pipeline
                if self.custom_name:
                    stop_pipeline(self.custom_name)
                    logger.info(f"Requested pipeline stop for: {self.custom_name}")
            except Exception as e:
                logger.error(f"Failed to stop pipeline: {e}")

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("Signal handlers registered for graceful cancellation")

    def discover_videos(self) -> List[str]:
        """
        Find all available raw videos for the tenant.

        Returns:
            List of video paths available for track generation.
        """
        try:
            storage = get_storage(self.tenant_id)
            path_manager = GCSPaths()

            available_videos = []

            # Look for videos in the raw_videos directory
            raw_videos_path = path_manager.get_path(
                "raw_data"
            )

            logger.info(f"Looking for videos in GCS path: {raw_videos_path}")

            try:
                available_videos = storage.list_blobs(
                    prefix=raw_videos_path
                )
                logger.info(f"Raw list_blobs result: {list(available_videos)}")
                # Filter for video files (common video extensions)
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
                available_videos = [
                    v for v in available_videos
                    if any(v.lower().endswith(ext) for ext in video_extensions)
                ]
                logger.info(f"Found {len(available_videos)} videos in raw_videos: {available_videos}")

            except Exception as e:
                logger.warning(f"Could not list videos from {raw_videos_path}: {e}")
                available_videos = []

            # Limit videos if specified
            if self.video_limit and self.video_limit > 0 and self.video_limit < len(available_videos):
                original_count = len(available_videos)
                available_videos = available_videos[:self.video_limit]
                logger.info(
                    f"🎯 LIMITED: Reduced from {original_count} to "
                    f"{self.video_limit} videos: {available_videos}"
                )
            elif self.video_limit and self.video_limit > 0:
                logger.info(
                    f"⚠️  Requested {self.video_limit} videos, but only "
                    f"{len(available_videos)} available"
                )
            else:
                logger.info(
                    f"📋 Using all {len(available_videos)} videos "
                    f"(video_limit={self.video_limit})"
                )

            return available_videos

        except Exception as e:
            logger.error(f"Error discovering videos: {e}")
            raise

    def _update_firestore_status(self, status: str, error: Optional[str] = None) -> None:
        """
        Update the Firestore document with the final track generation status.

        Args:
            status: Final status (completed, error, cancelled)
            error: Optional error message if status is error
        """
        if not self.firestore_client or not self.task_id:
            return

        try:
            from datetime import datetime, timezone

            doc_ref = self.firestore_client.collection("tracking_runs").document(self.task_id)

            update_data = {
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat() + "Z"
            }

            if error:
                update_data["error"] = error

            doc_ref.update(update_data)
            logger.info(f"Updated Firestore status for task_id {self.task_id} to: {status}")

        except Exception as e:
            logger.error(f"Failed to update Firestore status for task_id {self.task_id}: {e}")

    def execute(self) -> Dict[str, Any]:
        """Execute track generation for all discovered videos.

        Returns:
            Dictionary containing track generation results.
        """
        logger.info(f"--- Starting Track Generation Workflow Tenant: {self.tenant_id} ---")

        # Check for cancellation before starting
        if self.cancellation_event and self.cancellation_event.is_set():
            logger.info("Track generation cancelled before execution")
            result = {
                "status": "cancelled",
                "videos_found": 0,
                "videos_processed": 0,
                "run_id": self.custom_name,
                "pipeline_name": self.custom_name,
                "run_guids": [],
                "message": "Track generation cancelled before execution",
                "custom_name": self.custom_name,
                "total_runs": 0,
                "successful_runs": 0,
                "track_generation_results": [],
            }
            self._update_firestore_status("cancelled")
            return result

        try:
            videos = self.discover_videos()
            if not videos:
                logger.warning("No videos found for track generation")
                result = {
                    "status": "completed",
                    "videos_found": 0,
                    "videos_processed": 0,
                    "run_id": self.custom_name,
                    "pipeline_name": self.custom_name,
                    "run_guids": [],
                    "message": "No videos available for track generation",
                    "custom_name": self.custom_name,
                    "total_runs": 0,
                    "successful_runs": 0,
                    "track_generation_results": [],
                }
                self._update_firestore_status("completed")
                return result

            # Update status to running before starting
            if self.task_id:
                self._update_firestore_status("running")

            track_generation_results = []
            successful_runs = 0
            total_runs = len(videos)

            for video_path in videos:
                # Check for cancellation before processing each video
                if self.cancellation_event and self.cancellation_event.is_set():
                    logger.info("Track generation cancelled during video processing")
                    break

                try:
                    logger.info(f"Processing video: {video_path}")

                    # Create pipeline for this video
                    pipeline = TrackGeneratorPipeline(
                        config=self.detection_config,
                        tenant_id=self.tenant_id,
                        verbose=self.verbose,
                        save_intermediate=self.save_intermediate,
                        task_id=self.task_id
                    )

                    # Execute track generation
                    video_result = pipeline.run(
                        video_path=video_path,
                        resume_from_checkpoint=self.resume_from_checkpoint
                    )

                    track_generation_results.append({
                        "video_path": video_path,
                        "result": video_result
                    })

                    if video_result.get("status") == "completed":
                        successful_runs += 1
                        logger.info(f"Successfully processed video {video_path}")
                    else:
                        logger.warning(f"Failed to process video {video_path}: {video_result.get('status')}")

                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {e}")
                    track_generation_results.append({
                        "video_path": video_path,
                        "result": {
                            "status": "error",
                            "error": str(e)
                        }
                    })

            # Determine overall status
            if self.cancellation_event and self.cancellation_event.is_set():
                final_status = "cancelled"
                message = "Track generation cancelled during execution"
            elif successful_runs == total_runs:
                final_status = "completed"
                message = f"Successfully processed all {total_runs} videos"
            elif successful_runs > 0:
                final_status = "partial_success"
                message = f"Processed {successful_runs}/{total_runs} videos successfully"
            else:
                final_status = "failed"
                message = f"Failed to process any of the {total_runs} videos"

            result = {
                "status": final_status,
                "videos_found": len(videos),
                "videos_processed": len(track_generation_results),
                "run_id": self.custom_name,
                "pipeline_name": self.custom_name,
                "run_guids": [self.custom_name],
                "message": message,
                "custom_name": self.custom_name,
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "track_generation_results": track_generation_results,
            }

            self._update_firestore_status(final_status)
            return result

        except InterruptedError:
            logger.info("Track generation workflow cancelled")
            result = {
                "status": "cancelled",
                "videos_found": 0,
                "videos_processed": 0,
                "run_id": self.custom_name,
                "pipeline_name": self.custom_name,
                "run_guids": [self.custom_name],
                "custom_name": self.custom_name,
                "error": "Track generation workflow cancelled by user request",
                "total_runs": 0,
                "successful_runs": 0,
                "track_generation_results": [],
            }
            self._update_firestore_status("cancelled", "Track generation workflow cancelled by user request")
            return result
        except Exception as e:
            logger.error(f"Track generation workflow failed: {e}")
            result = {
                "status": "failed",
                "videos_found": 0,
                "videos_processed": 0,
                "run_id": self.custom_name,
                "pipeline_name": self.custom_name,
                "run_guids": [self.custom_name],
                "custom_name": self.custom_name,
                "error": str(e),
                "total_runs": 0,
                "successful_runs": 0,
                "track_generation_results": [],
            }
            self._update_firestore_status("error", str(e))
            return result


def create_unverified_tracks_workflow(tenant_id: str, **kwargs):
    """
    Convenience function for running track generation workflow.

    This function provides a simple interface for running the track generation workflow
    with keyword arguments.

    Args:
        tenant_id: The tenant ID for GCS operations.
        **kwargs: Additional keyword arguments for UnverifiedTrackGenerationWorkflow.

    Returns:
        Dictionary containing track generation results.
    """
    workflow = UnverifiedTrackGenerationWorkflow(tenant_id=tenant_id, **kwargs)
    return workflow.execute()
