import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from google.api_core.exceptions import GoogleAPIError

try:
    from google.cloud import pubsub_v1, firestore  # type: ignore
except (ImportError, ModuleNotFoundError) as import_error:  # pragma: no cover - dependency guard
    pubsub_v1 = None  # type: ignore[assignment]
    firestore = None  # type: ignore[assignment]
    _google_import_error = import_error
else:
    _google_import_error = None

from ..schemas.training import TrainingRequest, TrainingResponse, TrainingStatus  # type: ignore

logger = logging.getLogger(__name__)

JOB_NAME = "training-jobs"

router = APIRouter(prefix="/training", tags=["Training"])


class PubSubPublisher:
    """Pub/Sub publisher for training job requests."""

    def __init__(self):
        if pubsub_v1 is None or _google_import_error is not None:
            raise RuntimeError(
                "google-cloud-pubsub is required to publish training jobs. "
                "Install google-cloud-pubsub and ensure the environment is configured."
            ) from _google_import_error

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
        
        # Debug log the incoming request
        logger.info(f"📥 Received training request - tenant_id: {request.tenant_id}, custom_name: {request.custom_name}, dataset_address: {request.dataset_address}")
        logger.info(f"📥 Full request model: {request.model_dump(by_alias=False)}")

        # Convert structured params to dictionaries for Pub/Sub message
        training_params = request.training_params.model_dump(exclude_unset=True, by_alias=False) if request.training_params else {}
        model_params = request.model_params.model_dump(exclude_unset=True, by_alias=False) if request.model_params else {}
        eval_params = request.eval_params.model_dump(exclude_unset=True, by_alias=False) if request.eval_params else {}

        # Create message data
        message_data = {
            "action": "create",
            "task_id": task_id,
            "custom_name": request.custom_name,
            "tenant_id": request.tenant_id,
            "resume_from_checkpoint": request.resume_from_checkpoint,
            "dataset_address": request.dataset_address,
            "training_params": training_params,
            "model_params": model_params,
            "eval_params": eval_params,
            "timestamp": datetime.now(timezone.utc).isoformat()
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


class TrainingStatusManager:
    """Manages training job status queries from Firestore."""

    def __init__(self):
        if firestore is None or _google_import_error is not None:
            raise RuntimeError(
                "google-cloud-firestore is required to query training status. "
                "Install google-cloud-firestore and ensure the environment is configured."
            ) from _google_import_error

        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
        self.db = firestore.Client(project=self.project_id)
        self._runs_collection = self.db.collection("training_runs")

    def get_training_job_status(self, task_id: str) -> dict:
        """Get status for a specific training job."""
        try:
            doc = self._runs_collection.document(task_id).get()
            if not doc.exists:
                return {"task_id": task_id, "status": "not_found", "error": "No training job found with this task_id."}
            data = doc.to_dict() or {}
            return data
        except Exception as e:
            logger.error(f"Error retrieving training job status for {task_id}: {e}")
            return {"task_id": task_id, "status": "error", "error": str(e)}

    def list_training_jobs(self, limit: int = 50) -> List[dict]:
        """List all training jobs, ordered by creation time (newest first)."""
        
        # Validate limit parameter
        if limit < 1:
            limit = 1
        elif limit > 100:
            limit = 100
            
        try:
            # Query Firestore for all training runs, ordered by created_at descending
            query = self._runs_collection.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)  # type: ignore
            docs = query.stream()

            jobs = []
            for doc in docs:
                data = doc.to_dict() or {}
                data["task_id"] = doc.id  # Add the document ID as task_id
                jobs.append(data)

            return jobs
        except Exception as e:
            logger.error(f"Error listing training jobs: {e}")
            return []

    def list_training_jobs_by_tenant(self, tenant_id: str, status_filter: Optional[str] = None, limit: int = 50) -> List[dict]:
        """List training jobs for a specific tenant, optionally filtered by status."""
        
        # Validate limit parameter
        if limit < 1:
            limit = 1
        elif limit > 100:
            limit = 100
            
        try:
            # Start with base query for the tenant
            query = self._runs_collection.where("tenant_id", "==", tenant_id)
            
            # Add status filter if provided
            if status_filter:
                query = query.where("status", "==", status_filter)
            
            # Order by creation time (newest first) and limit results
            query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)  # type: ignore
            docs = query.stream()

            jobs = []
            for doc in docs:
                data = doc.to_dict() or {}
                data["task_id"] = doc.id  # Add the document ID as task_id
                jobs.append(data)

            return jobs
        except Exception as e:
            logger.error(f"Error listing training jobs for tenant {tenant_id}: {e}")
            return []


# Global instances (lazily instantiated to avoid import failures during tests)
publisher: Optional[PubSubPublisher]
status_manager: Optional[TrainingStatusManager]

if _google_import_error is None:
    publisher = PubSubPublisher()
    status_manager = TrainingStatusManager()
else:  # pragma: no cover - executed only when dependency missing
    publisher = None
    status_manager = None


@router.post("", response_model=TrainingResponse)
async def start_training(request: TrainingRequest) -> TrainingResponse:
    """Queue a new training job."""

    try:
        # Publish to Pub/Sub
        if publisher is None:
            raise HTTPException(
                status_code=503,
                detail="Training service dependencies are unavailable. Please install google-cloud-pubsub and retry.",
            )

        task_id = publisher.publish_training_request(request)

        return TrainingResponse(
            task_id=task_id,
            status="queued",
            message="Training job has been queued successfully",
            created_at=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(f"Error queuing training job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/")
async def list_training_jobs(limit: int = 50):
    """List all training jobs, ordered by creation time (newest first)."""

    # Validate limit parameter
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")

    try:
        if status_manager is None:
            raise HTTPException(
                status_code=503,
                detail="Training status dependencies are unavailable. Please install google-cloud-firestore and retry.",
            )

        jobs = status_manager.list_training_jobs(limit=limit)
        return {
            "jobs": jobs,
            "count": len(jobs),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training jobs")


@router.get("/tenant/{tenant_id}")
async def list_training_jobs_by_tenant(
    tenant_id: str, 
    status: Optional[str] = None, 
    limit: int = 50
):
    """List training jobs for a specific tenant, optionally filtered by status."""

    # Validate limit parameter
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")

    if status_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Training status dependencies are unavailable. Please install google-cloud-firestore and retry.",
        )

    try:
        jobs = status_manager.list_training_jobs_by_tenant(
            tenant_id=tenant_id, 
            status_filter=status, 
            limit=limit
        )
        return {
            "tenant_id": tenant_id,
            "jobs": jobs,
            "count": len(jobs),
            "limit": limit,
            "status_filter": status
        }
    except Exception as e:
        logger.error(f"Error listing training jobs for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training jobs")


@router.get("/{task_id}", response_model=TrainingStatus)
async def get_training_job_status(task_id: str):
    """Get the status of a specific training job."""

    # Validate task_id format (should be a valid UUID)
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format. Must be a valid UUID.")

    if status_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Training status dependencies are unavailable. Please install google-cloud-firestore and retry.",
        )

    try:
        status_data = status_manager.get_training_job_status(task_id)

        # Check if job was not found
        if status_data.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Training job {task_id} not found")

        # Check if there was an error retrieving status
        if status_data.get("status") == "error":
            raise HTTPException(status_code=500, detail=status_data.get("error", "Unknown error"))

        return TrainingStatus(**status_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving training job status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training job status")


@router.delete("/{task_id}")
async def cancel_training_job(task_id: str):
    """Cancel a training job by publishing a cancel message."""

    # Validate task_id format (should be a valid UUID)
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format. Must be a valid UUID.")

    if publisher is None:
        raise HTTPException(
            status_code=503,
            detail="Training service dependencies are unavailable. Please install google-cloud-pubsub and retry.",
        )

    try:
        # Create cancel message
        message_data = {
            "action": "cancel",
            # Standardize on task_id; the proxy will map this to the Cloud Run execution/operation
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
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
