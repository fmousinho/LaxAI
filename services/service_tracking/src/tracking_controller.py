import logging
logger = logging.getLogger(__name__)

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.utils.id_generator import create_simple_uuid
from schemas.tracking import TrackingParams
from shared_libs.common.detection import DetectionModel

DEFAULT_CONFIDENCE_THRESHOLD = 0.4

class TrackingController:
    """
    Orchestrates the end-to-end unverified track generation process.
    Handles video discovery, pipeline execution, and status reporting.
    """

    def __init__(
        self,
        tenant_id: str,
        tracking_params: TrackingParams,
        wandb_run_name: str = "track_generation_run",
        task_id: Optional[str] = None,

    ):
        self.tenant_id = tenant_id
        self.tracking_params = tracking_params
        self.wandb_run_name = wandb_run_name
        self.task_id = task_id

        self.storage_client = get_storage(tenant_id)
        self.path_manager = GCSPaths()

        self.detector = DetectionModel()
        self.tracker = self._prepare_tracker()

  

    def _prepare_tracker(self):
        """Loads tracking model based on tracking parameters."""
        return TrackingModel()







    def discover_videos(self) -> List[str]:
        """Find all available raw videos for the tenant."""
        try:
            raw_videos_path = self.path_manager.get_path("raw_data")
            logger.info(f"Looking for videos in GCS path: {raw_videos_path}")

            blobs = self.storage_client.list_blobs(prefix=raw_videos_path)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            available_videos = [
                b for b in blobs
                if any(b.lower().endswith(ext) for ext in video_extensions)
            ]
            
            logger.info(f"Found {len(available_videos)} videos")

            # Apply limit if specified
            limit = self.tracking_params.video_limit
            if limit and limit > 0:
                available_videos = available_videos[:limit]
                logger.info(f"Limited to {limit} videos")

            return available_videos

        except Exception as e:
            logger.error(f"Error discovering videos: {e}")
            raise

    

    def run(self) -> Dict[str, Any]:
        """Execute the tracking process."""
        logger.info(f"Starting Tracking Controller for tenant: {self.tenant_id}")
        
        if self.cancellation_event.is_set():
            return {"status": "cancelled", "message": "Cancelled before start"}

        try:
            videos = self.discover_videos()
            if not videos:
                self._update_firestore_status("completed")
                return {"status": "completed", "message": "No videos found"}

            results = []
            successful_runs = 0
            total_runs = len(videos)

            for i, video_path in enumerate(videos):
                if self.cancellation_event.is_set():
                    break

                try:
                    logger.info(f"Processing video {i+1}/{total_runs}: {video_path}")
                    
                    # Initialize pipeline with new params
                    pipeline = TrackGeneratorPipeline(
                        tracking_params=self.tracking_params,
                        tenant_id=self.tenant_id,
                        verbose=self.verbose,
                        task_id=self.task_id
                    )

                    video_result = pipeline.run(
                        video_path=video_path,
                        resume_from_checkpoint=self.tracking_params.resume_from_checkpoint
                    )

                    results.append({"video_path": video_path, "result": video_result})
                    
                    if video_result.get("status") == "completed":
                        successful_runs += 1
                    
                    self._update_firestore_progress(i + 1, total_runs)

                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {e}")
                    results.append({
                        "video_path": video_path, 
                        "result": {"status": "error", "error": str(e)}
                    })

            # Determine final status
            if self.cancellation_event.is_set():
                status = "cancelled"
                message = "Cancelled during execution"
            elif successful_runs == total_runs:
                status = "completed"
                message = "All videos processed successfully"
            elif successful_runs > 0:
                status = "completed" # Partial success is still completed
                message = f"Processed {successful_runs}/{total_runs} videos"
            else:
                status = "error"
                message = "All videos failed"

            self._update_firestore_status(status, message if status == "error" else None)
            
            return {
                "status": status,
                "message": message,
                "results": results,
                "successful_runs": successful_runs,
                "total_runs": total_runs
            }

        except Exception as e:
            logger.error(f"Tracking run failed: {e}")
            self._update_firestore_status("error", str(e))
            raise
