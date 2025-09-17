import json
import logging
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from google.api_core.exceptions import GoogleAPIError
from google.cloud import pubsub_v1  # type: ignore

from ..schemas.training import TrainingRequest, TrainingResponse

logger = logging.getLogger(__name__)

JOB_NAME = "training-jobs"

router = APIRouter(prefix="/train", tags=["training"])


class PubSubPublisher:
    """Pub/Sub publisher for training job requests."""

    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
        self.topic_name = JOB_NAME
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(self.project_id, self.topic_name)

    def publish_training_request(self, request: TrainingRequest) -> str:
        """Publish a training request to Pub/Sub."""
        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Convert structured params to dictionaries for Pub/Sub message
        training_params = request.training_params.model_dump(exclude_unset=True) if request.training_params else {}
        model_params = request.model_params.model_dump(exclude_unset=True) if request.model_params else {}
        eval_params = request.eval_params.model_dump(exclude_unset=True) if request.eval_params else {}

        # Create message data
        message_data = {
            "action": "create",
            "task_id": task_id,
            "custom_name": request.custom_name,
            "tenant_id": request.tenant_id,
            "resume_from_checkpoint": request.resume_from_checkpoint,
            "training_params": training_params,
            "model_params": model_params,
            "eval_params": eval_params,
            "timestamp": datetime.now().isoformat()
        }

        # Convert to JSON bytes
        data = json.dumps(message_data).encode('utf-8')

        try:
            # Publish message
            future = self.publisher.publish(self.topic_path, data)
            message_id = future.result()

            logger.info(f"Published training request {task_id} to Pub/Sub (message_id: {message_id})")
            logger.info(f"Message data: {message_data}")
            return task_id

        except GoogleAPIError as e:
            logger.error(f"Failed to publish training request: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to queue training job: {str(e)}")


# Global publisher instance
publisher = PubSubPublisher()


@router.post("/", response_model=TrainingResponse)
async def start_training(request: TrainingRequest) -> TrainingResponse:
    """Queue a new training job."""

    try:
        # Publish to Pub/Sub
        task_id = publisher.publish_training_request(request)

        return TrainingResponse(
            task_id=task_id,
            status="queued",
            message="Training job has been queued successfully",
            created_at=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error queuing training job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/")
async def list_training_jobs():
    """List all queued training jobs."""

    # Note: This is a simplified implementation
    # In a real system, you might want to store job status in Firestore or similar
    return {
        "message": "Job listing not implemented in API service. Use training service for job status.",
        "jobs": []
    }


@router.delete("/{task_id}")
async def cancel_training_job(task_id: str):
    """Cancel a training job by publishing a cancel message."""

    try:
        # Create cancel message
        message_data = {
            "action": "cancel",
            "job_id": task_id,  # This should match the job name created by the proxy
            "timestamp": datetime.utcnow().isoformat()
        }

        # Convert to JSON bytes
        data = json.dumps(message_data).encode('utf-8')

        # Publish cancel message
        future = publisher.publisher.publish(publisher.topic_path, data)
        message_id = future.result()

        logger.info(f"Published cancel request for job {task_id} to Pub/Sub (message_id: {message_id})")

        return {
            "message": "Cancel request has been queued",
            "task_id": task_id
        }

    except GoogleAPIError as e:
        logger.error(f"Failed to publish cancel request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue cancel request: {str(e)}")
    except Exception as e:
        logger.error(f"Error cancelling training job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
