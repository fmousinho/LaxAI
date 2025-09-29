import json
import logging
import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from google.api_core.exceptions import GoogleAPIError
from google.cloud import pubsub_v1, firestore  # type: ignore

from ..schemas.tracking import TrackingRequest, TrackingResponse, TrackingStatus

logger = logging.getLogger(__name__)

JOB_NAME = "tracking-jobs"

router = APIRouter(prefix="/track", tags=["tracking"])


class PubSubPublisher:
    """Pub/Sub publisher for tracking job requests."""

    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
        self.topic_name = JOB_NAME
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(self.project_id, self.topic_name)

    def publish_tracking_request(self, request: TrackingRequest) -> str:
        """Publish a tracking request to Pub/Sub."""
        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Convert structured params to dictionaries for Pub/Sub message
        tracking_params = request.tracking_params.model_dump(exclude_unset=True) if request.tracking_params else {}

        # Create message data
        message_data = {
            "action": "create",
            "task_id": task_id,
            "custom_name": request.custom_name,
            "tenant_id": request.tenant_id,
            "tracking_params": tracking_params,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Convert to JSON bytes
        data = json.dumps(message_data).encode('utf-8')

        try:
            # Publish message
            future = self.publisher.publish(self.topic_path, data)
            message_id = future.result()

            logger.info(f"Published tracking request {task_id} to Pub/Sub (message_id: {message_id})")
            logger.info(f"Message data: {message_data}")
            return task_id

        except GoogleAPIError as e:
            logger.error(f"Failed to publish tracking request: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to queue tracking job: {str(e)}")


class TrackingStatusManager:
    """Manages tracking job status queries from Firestore."""

    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
        self.db = firestore.Client(project=self.project_id)
        self._runs_collection = self.db.collection("tracking_runs")

    def get_tracking_job_status(self, task_id: str) -> dict:
        """Get status for a specific tracking job."""
        try:
            doc = self._runs_collection.document(task_id).get()
            if not doc.exists:
                return {"task_id": task_id, "status": "not_found", "error": "No tracking job found with this task_id."}
            data = doc.to_dict() or {}
            return data
        except Exception as e:
            logger.error(f"Error retrieving tracking job status for {task_id}: {e}")
            return {"task_id": task_id, "status": "error", "error": str(e)}

    def list_tracking_jobs(self, limit: int = 50) -> List[dict]:
        """List all tracking jobs, ordered by creation time (newest first)."""

        # Validate limit parameter
        if limit < 1:
            limit = 1
        elif limit > 100:
            limit = 100

        try:
            # Query Firestore for tracking runs, ordered by creation time descending
            query = self._runs_collection.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
            docs = query.stream()

            jobs = []
            for doc in docs:
                data = doc.to_dict()
                if data:
                    jobs.append(data)

            return jobs

        except Exception as e:
            logger.error(f"Error listing tracking jobs: {e}")
            return []


# Global instances
_publisher = None
_status_manager = None


def get_publisher() -> PubSubPublisher:
    """Get or create PubSubPublisher instance."""
    global _publisher
    if _publisher is None:
        _publisher = PubSubPublisher()
    return _publisher


def get_status_manager() -> TrackingStatusManager:
    """Get or create TrackingStatusManager instance."""
    global _status_manager
    if _status_manager is None:
        _status_manager = TrackingStatusManager()
    return _status_manager


@router.post("", response_model=TrackingResponse)
async def create_tracking_job(request: TrackingRequest):
    """
    Create a new tracking job.

    This endpoint publishes a tracking request to Pub/Sub, which triggers
    the tracking proxy Cloud Function to execute the job via Cloud Run.
    """
    try:
        publisher = get_publisher()
        task_id = publisher.publish_tracking_request(request)

        return TrackingResponse(
            task_id=task_id,
            status="queued",
            message="Tracking job queued successfully",
            created_at=datetime.now(timezone.utc).isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating tracking job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/")
async def list_tracking_jobs(limit: int = 50):
    """
    List all tracking jobs.

    Returns a list of tracking jobs ordered by creation time (newest first).
    """
    try:
        status_manager = get_status_manager()
        jobs = status_manager.list_tracking_jobs(limit=limit)
        return {"jobs": jobs, "count": len(jobs)}

    except Exception as e:
        logger.error(f"Error listing tracking jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list tracking jobs")


@router.get("/{task_id}", response_model=TrackingStatus)
async def get_tracking_job_status(task_id: str):
    """
    Get the status of a specific tracking job.

    Returns detailed status information for the tracking job identified by task_id.
    """
    try:
        status_manager = get_status_manager()
        status_data = status_manager.get_tracking_job_status(task_id)

        # Convert to TrackingStatus model
        return TrackingStatus(**status_data)

    except Exception as e:
        logger.error(f"Error getting tracking job status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status for task {task_id}")


@router.delete("/{task_id}")
async def cancel_tracking_job(task_id: str):
    """
    Cancel a tracking job.

    This endpoint sends a cancellation request to the tracking proxy.
    Note: This requires the tracking proxy to support cancellation.
    """
    try:
        # For now, we'll just update the status to cancelled in Firestore
        # In a full implementation, this would send a message to cancel the Cloud Run job
        status_manager = get_status_manager()

        # Get current status
        current_status = status_manager.get_tracking_job_status(task_id)
        if current_status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Tracking job {task_id} not found")

        # Update status to cancelled
        update_data = {
            "status": "cancelled",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": "Job cancelled by user"
        }

        status_manager._runs_collection.document(task_id).update(update_data)

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Tracking job cancellation requested"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling tracking job {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel tracking job {task_id}")


@router.get("/{task_id}/progress", response_model=Dict[str, Any])
async def get_tracking_progress(task_id: str):
    """
    Get current progress for a tracking job.

    Returns the latest processing status, frames processed, and other progress metrics.
    """
    try:
        # Get progress from Firestore
        db = firestore.Client()
        progress_doc = db.collection('tracking_progress').document(task_id).get()
        
        if not progress_doc.exists:
            raise HTTPException(status_code=404, detail=f"No progress data found for task {task_id}")
        
        progress_data = progress_doc.to_dict()
        if not progress_data:
            raise HTTPException(status_code=404, detail=f"No progress data found for task {task_id}")
            
        return progress_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting progress for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress for task {task_id}")


@router.get("/{task_id}/progress/stream")
async def stream_tracking_progress(task_id: str):
    """
    Server-Sent Events endpoint for real-time progress updates.

    Returns a stream of progress updates for authenticated clients.
    """
    async def generate():
        try:
            db = firestore.Client()
            last_update = None
            
            while True:
                # Get latest progress
                progress_doc = db.collection('tracking_progress').document(task_id).get()
                
                if progress_doc.exists:
                    progress_data = progress_doc.to_dict()
                    
                    if progress_data:
                        updated_at = progress_data.get('updated_at')
                        
                        # Only send if there's new data
                        if updated_at != last_update:
                            last_update = updated_at
                            yield f"data: {json.dumps(progress_data)}\n\n"
                            
                            # If job is complete, end the stream
                            if progress_data.get('status') in ['completed', 'failed', 'cancelled']:
                                break
                
                # Wait before checking again
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Error in progress stream for task {task_id}: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )