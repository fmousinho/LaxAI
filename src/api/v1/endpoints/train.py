"""
Training API endpoint for LaxAI.
"""
import asyncio
import logging
import traceback
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ..schemas.training import (
    TrainingRequest, 
    TrainingResponse, 
    TrainingProgress, 
    ErrorResponse
)
from common.pipeline import get_active_pipelines
from services.training_service import (
    create_job,
    start_job,
    get_job,
    list_jobs,
    cancel_job
)

logger = logging.getLogger(__name__)

# The service manages job storage now

router = APIRouter()


# Request conversion is handled inside the service


# Background execution handled by service


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a training job with the provided configuration.
    
    Returns a task ID that can be used to track progress.
    """
    try:
        # Create job entry and get kwargs
        task_id, kwargs = create_job(request)

        # Schedule the job via the service
        start_job(task_id, kwargs, background_tasks)

        logger.info(f"Started training task {task_id} for tenant {request.tenant_id}")

        return TrainingResponse(
            status="accepted",
            task_id=task_id,
            message=f"Training job started with ID: {task_id}"
        )

    except Exception as e:
        error_msg = f"Failed to start training: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Error details: {traceback.format_exc()}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail=error_msg,
                error_type="training_start_failed"
            ).model_dump()
        )


@router.get("/train/{task_id}/status", response_model=TrainingResponse)
async def get_training_status(task_id: str):
    """
    Get the status of a training job.
    """
    job = get_job(task_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                detail=f"Training task {task_id} not found",
                error_type="task_not_found"
            ).model_dump()
        )
    return TrainingResponse(
        status=job["status"],
        task_id=task_id,
        message=f"Training job {task_id} is {job['status']}"
    )


@router.get("/train/jobs", response_model=Dict[str, Any])
async def list_training_jobs():
    """
    List all training jobs and their statuses.
    """
    jobs = list_jobs()
    jobs_summary = {}
    for task_id, job in jobs.items():
        jobs_summary[task_id] = {
            "status": job["status"],
            "tenant_id": job["request"]["tenant_id"],
            "custom_name": job["request"]["custom_name"],
            "created_at": job["created_at"],
            "progress_status": job["progress"]["status"],
            "progress_message": job["progress"]["message"]
        }

    return {"total_jobs": len(jobs), "jobs": jobs_summary}


@router.delete("/train/{task_id}")
async def cancel_training_job(task_id: str):
    """
    Cancel a training job using the pipeline control system.
    """
    job = get_job(task_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                detail=f"Training task {task_id} not found",
                error_type="task_not_found"
            ).model_dump()
        )

    pipeline_stopped = cancel_job(task_id)

    return {"message": f"Training job {task_id} cancelled", "pipeline_stopped": pipeline_stopped}


@router.get("/train/pipelines", response_model=Dict[str, Any])
async def list_active_pipelines():
    """
    List all currently active training pipelines.
    """
    active_pipelines = get_active_pipelines()
    
    return {
        "total_active_pipelines": len(active_pipelines),
        "pipelines": active_pipelines
    }


@router.delete("/train/pipelines/{pipeline_name}")
async def stop_training_pipeline(pipeline_name: str):
    """
    Stop a specific training pipeline by name.
    """
    pipeline_stopped = stop_pipeline(pipeline_name)
    
    if pipeline_stopped:
        return {"message": f"Pipeline '{pipeline_name}' stop requested successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                detail=f"Pipeline '{pipeline_name}' not found or not active",
                error_type="pipeline_not_found"
            ).model_dump()
        )
