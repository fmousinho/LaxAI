"""Training API endpoints - Proxy to service_training."""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException

try:
    from google.cloud import firestore  # type: ignore
except (ImportError, ModuleNotFoundError) as import_error:  # pragma: no cover
    firestore = None  # type: ignore[assignment]
    _firestore_import_error = import_error
else:
    _firestore_import_error = None

# Import schemas from service_training (single source of truth)
from services.service_training.src.schemas.training import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
)
from services.service_training.src.schemas.firestore import TrainingJobDocument
from services.service_training.src.state_machine import TrainingJobState
from services.service_training.src.state_manager import TrainingJobStateManager
from services.service_training.src.schemas.pubsub import AutoResumeMessage, CancellationMessage
from services.service_training.src.config.pubsub_config import PubSubConfig

try:
    from google.cloud import pubsub_v1
except (ImportError, ModuleNotFoundError):
    pubsub_v1 = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["Training"])

# Initialize state manager and Pub/Sub config
state_manager = TrainingJobStateManager()
pubsub_config = PubSubConfig()


@router.post("", response_model=TrainingResponse)
async def start_training(request: TrainingRequest) -> TrainingResponse:
    """Queue a new training job by creating Firestore document and publishing to Pub/Sub."""
    
    if pubsub_v1 is None:
        raise HTTPException(
            status_code=503,
            detail="Training service dependencies unavailable. Install google-cloud-pubsub.",
        )

    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create Firestore document for the job
        job = TrainingJobDocument(
            task_id=task_id,
            tenant_id=request.tenant_id,
            status=TrainingJobState.QUEUED,
            wandb_run_name=request.wandb_run_name,
            training_params=request.training_params.dict() if request.training_params else {},
            eval_params=request.eval_params.dict() if request.eval_params else {},
            dataset_address=request.training_params.dataset_address if request.training_params else None,
            total_epochs=request.training_params.num_epochs if request.training_params else None,
        )
        
        # Store in Firestore
        if not state_manager.create_job(job):
            raise HTTPException(status_code=500, detail="Failed to create job in Firestore")
        
        # Publish to Pub/Sub for processing
        try:
            publisher = pubsub_v1.PublisherClient()
            topic_path = pubsub_config.get_topic_path(publisher)
            
            # Create message payload
            # We use AutoResumeMessage as the standard start message
            # It contains the task_id which is enough for the worker to fetch details from Firestore
            message = AutoResumeMessage(
                task_id=task_id,
                tenant_id=request.tenant_id,
                wandb_run_name=request.wandb_run_name,
                training_params=request.training_params.dict() if request.training_params else {},
                eval_params=request.eval_params.dict() if request.eval_params else {},
                dataset_address=request.training_params.dataset_address if request.training_params else None,
                auto_resume_count=0
            )
            
            # Publish message
            future = publisher.publish(
                topic_path, 
                message.to_json().encode('utf-8'),
                tenant_id=request.tenant_id,  # Add attributes for filtering if needed
                type="training_start"
            )
            message_id = future.result()
            logger.info(f"Published training start message for task {task_id}, message_id: {message_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish to Pub/Sub: {e}")
            # We still return success because the job is in Firestore, 
            # but we should probably alert or handle this better in production
            # For now, we'll log it and proceed

        
        return TrainingResponse(
            task_id=task_id,
            status=TrainingJobState.QUEUED.value,
            message="Training job has been queued successfully",
            created_at=job.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing training job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue training job: {str(e)}")


@router.get("/{task_id}", response_model=TrainingStatus)
async def get_training_job_status(task_id: str) -> TrainingStatus:
    """Get the status of a specific training job from Firestore."""
    
    # Validate task_id format
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format. Must be a valid UUID.")
    
    job = state_manager.get_job(task_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {task_id} not found")
    
    return TrainingStatus(
        task_id=job.task_id,
        status=job.status.value,
        progress=job.progress,
        current_epoch=None,  # Retrieved from checkpoint data
        total_epochs=job.total_epochs,
        loss=None,  # Tracked in metrics module
        metrics=job.metrics,
        logs=job.logs,
        updated_at=job.updated_at.isoformat()
    )


@router.delete("/{task_id}")
async def cancel_training_job(task_id: str):
    """Cancel a training job by updating Firestore and publishing cancellation message."""
    
    # Validate task_id format
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format. Must be a valid UUID.")
    
    job = state_manager.get_job(task_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {task_id} not found")
    
    # Update status to cancelled (with validation)
    from services.service_training.src.state_machine import StateTransition
    
    if StateTransition.is_terminal_state(job.status):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in terminal state '{job.status.value}'"
        )
    
    success = state_manager.update_status(task_id, TrainingJobState.CANCELLED)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state transition from '{job.status.value}' to 'cancelled'"
        )
    
    # Optionally publish cancellation message to Pub/Sub
    # This would notify the training worker to stop gracefully
    if pubsub_v1:
        try:
            publisher = pubsub_v1.PublisherClient()
            topic_path = pubsub_config.get_topic_path(publisher)
            
            cancel_msg = CancellationMessage(
                task_id=task_id,
                reason="User requested cancellation"
            )
            
            future = publisher.publish(topic_path, cancel_msg.to_json().encode('utf-8'))
            future.result()  # Wait for publish
            logger.info(f"Published cancellation message for task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to publish cancellation message: {e}")
            # Don't fail the request if Pub/Sub fails
    
    return {
        "message": "Training job cancelled successfully",
        "task_id": task_id,
        "status": "cancelled"
    }


@router.get("/")
async def list_training_jobs(limit: int = 50):
    """List all training jobs from Firestore."""
    
    # Validate limit
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    if firestore is None or _firestore_import_error:
        raise HTTPException(
            status_code=503,
            detail="Firestore unavailable. Install google-cloud-firestore.",
        )
    
    try:
        db = firestore.Client()
        collection = db.collection("training_runs")
        
        # Query Firestore for training runs, ordered by created_at descending
        query = collection.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)  # type: ignore
        docs = query.stream()
        
        jobs = []
        for doc in docs:
            data = doc.to_dict() or {}
            jobs.append({
                "task_id": doc.id,
                "tenant_id": data.get("tenant_id"),
                "wandb_run_name": data.get("wandb_run_name"),
                "status": data.get("status"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "progress": data.get("progress"),
            })
        
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
    
    # Validate limit
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    if firestore is None or _firestore_import_error:
        raise HTTPException(
            status_code=503,
            detail="Firestore unavailable. Install google-cloud-firestore.",
        )
    
    try:
        db = firestore.Client()
        collection = db.collection("training_runs")
        
        # Build query
        query = collection.where("tenant_id", "==", tenant_id)
        
        if status:
            query = query.where("status", "==", status)
        
        query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)  # type: ignore
        docs = query.stream()
        
        jobs = []
        for doc in docs:
            data = doc.to_dict() or {}
            jobs.append({
                "task_id": doc.id,
                "tenant_id": data.get("tenant_id"),
                "wandb_run_name": data.get("wandb_run_name"),
                "status": data.get("status"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "progress": data.get("progress"),
            })
        
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
