"""
Cloud training endpoints for job orchestration and monitoring.
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from api.v1.schemas.training import TrainingRequest
from cloud.firestore_client import JobStatus, get_firestore_client
from cloud.function_orchestrator import (publish_job_to_pubsub,
                                         validate_request_payload)
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cloud", tags=["cloud"])


class JobSubmissionResponse(BaseModel):
    """Response model for job submission."""
    task_id: str
    status: str
    message: str
    estimated_duration: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    task_id: str
    status: str
    progress: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobListItem(BaseModel):
    """Response model for job list item."""
    task_id: str
    status: str
    tenant_id: str
    created_at: datetime
    updated_at: datetime
    progress_percent: Optional[float] = None


class JobListResponse(BaseModel):
    """Response model for job list."""
    jobs: List[JobListItem]
    total_count: int
    page: int
    page_size: int


@router.post("/jobs/submit", response_model=JobSubmissionResponse)
async def submit_training_job(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> JobSubmissionResponse:
    """
    Submit a training job to the cloud queue.
    
    Args:
        request: Training job configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Job submission response with task ID and status
    """
    try:
        logger.info(f"Submitting training job for tenant {request.tenant_id}")
        
        # Generate unique task ID
        task_id = f"training_job_{uuid.uuid4().hex[:12]}"
        
        # Validate the request
        is_valid, error_msg, validated_request = validate_request_payload(request.dict())
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request: {error_msg}"
            )
        
        # Get Firestore client and create job document
        firestore_client = get_firestore_client()
        if not firestore_client:
            raise HTTPException(
                status_code=503,
                detail="Firestore client not available"
            )
        
        # Create job document in Firestore
        firestore_client.create_job(
            task_id=task_id,
            request_payload=request.dict(),
            metadata={"api_endpoint": "fastapi", "created_by": "cloud_endpoint"}
        )
        
        # Publish to Pub/Sub for worker processing
        success = publish_job_to_pubsub(task_id)
        if not success:
            # Update job status to failed if pub/sub fails
            firestore_client.update_job_status(
                task_id, 
                JobStatus.FAILED, 
                error_details={"error": "Failed to publish to Pub/Sub"}
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to queue job for processing"
            )
        
        return JobSubmissionResponse(
            task_id=task_id,
            status=JobStatus.QUEUED.value,
            message="Training job submitted successfully",
            estimated_duration="30-60 minutes"
        )
        
    except Exception as e:
        logger.error(f"Failed to submit training job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit training job: {str(e)}"
        )


@router.get("/jobs/{task_id}/status", response_model=JobStatusResponse)
async def get_job_status(task_id: str) -> JobStatusResponse:
    """
    Get the status of a training job.
    
    Args:
        task_id: Job identifier
        
    Returns:
        Current job status and progress
    """
    try:
        # Get Firestore client
        firestore_client = get_firestore_client()
        if not firestore_client:
            raise HTTPException(
                status_code=503,
                detail="Firestore client not available"
            )
        
        # Retrieve job from Firestore
        job_data = firestore_client.get_job(task_id)
        if not job_data:
            raise HTTPException(
                status_code=404,
                detail=f"Job {task_id} not found"
            )
        
        return JobStatusResponse(
            task_id=task_id,
            status=job_data.get("status", "QUEUED"),
            progress=job_data.get("progress", {}),
            created_at=job_data.get("created_at", datetime.utcnow()),
            updated_at=job_data.get("updated_at", datetime.utcnow()),
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.delete("/jobs/{task_id}")
async def cancel_job(task_id: str) -> Dict[str, Any]:
    """
    Cancel a training job.
    
    Args:
        task_id: Job identifier
        
    Returns:
        Cancellation confirmation
    """
    try:
        # Get Firestore client
        firestore_client = get_firestore_client()
        if not firestore_client:
            raise HTTPException(
                status_code=503,
                detail="Firestore client not available"
            )
        
        # First try to stop the actual pipeline if it's running locally
        logger.info(f"CloudAPI: Attempting to cancel job {task_id} - checking for local pipeline")
        from training_service import cancel_job as stop_pipeline_job
        pipeline_stopped = stop_pipeline_job(task_id)
        logger.info(f"CloudAPI: Local pipeline stop result for job {task_id}: {pipeline_stopped}")
        
        # Then update job status to cancelled in Firestore
        logger.info(f"CloudAPI: Updating Firestore status for cancelled job {task_id}")
        success = firestore_client.update_job_status(
            task_id, 
            JobStatus.CANCELLED,
            progress={"message": "Job cancelled by user request", "pipeline_stopped": pipeline_stopped}
        )
        
        if not success:
            logger.error(f"CloudAPI: Failed to update Firestore status for cancelled job {task_id}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to cancel job {task_id}"
            )
        
        logger.info(f"CloudAPI: Successfully cancelled job {task_id} (pipeline_stopped: {pipeline_stopped})")
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Job cancelled successfully",
            "pipeline_stopped": pipeline_stopped
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of jobs per page")
) -> JobListResponse:
    """
    List training jobs with optional filtering and pagination.
    
    Args:
        status: Optional status filter
        page: Page number (1-based)
        page_size: Number of jobs per page (max 100)
        
    Returns:
        Paginated list of jobs
    """
    try:
        # Get Firestore client
        firestore_client = get_firestore_client()
        if not firestore_client:
            raise HTTPException(
                status_code=503,
                detail="Firestore client not available"
            )
        
        # Convert status string to JobStatus enum if provided
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status.upper())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: {[s.value for s in JobStatus]}"
                )
        
        # Calculate limit based on pagination
        # Note: Current Firestore client doesn't support offset, so we'll get more and slice
        limit = page * page_size
        
        # Get jobs from Firestore
        all_jobs = firestore_client.list_jobs(
            status_filter=status_filter,
            limit=limit
        )
        
        # Calculate pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_jobs = all_jobs[start_idx:end_idx]
        
        # Convert to response format
        jobs = []
        for job_data in paginated_jobs:
            progress = job_data.get("progress", {})
            request_payload = job_data.get("request_payload", {})
            
            jobs.append(JobListItem(
                task_id=job_data["task_id"],
                status=job_data.get("status", "unknown"),
                tenant_id=request_payload.get("tenant_id", "unknown"),
                created_at=job_data.get("created_at", datetime.utcnow()),
                updated_at=job_data.get("updated_at", datetime.utcnow()),
                progress_percent=progress.get("progress_percent")
            ))
        
        return JobListResponse(
            jobs=jobs,
            total_count=len(all_jobs),  # This is approximate for the filtered set
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get("/health")
async def cloud_health_check() -> Dict[str, str]:
    """
    Health check for cloud components.
    
    Returns:
        Health status of cloud services
    """
    try:
        # Check Firestore connectivity
        firestore_client = get_firestore_client()
        firestore_status = "available" if firestore_client else "unavailable"
        
        # Check pub/sub (can't easily test without making a request, so assume available)
        pubsub_status = "available"
        
        return {
            "status": "healthy",
            "firestore": firestore_status,
            "pubsub": pubsub_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cloud health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
