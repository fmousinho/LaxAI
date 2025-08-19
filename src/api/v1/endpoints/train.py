"""
Training API endpoint for LaxAI.
"""
import asyncio
import logging
import traceback
import uuid
import functools
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ..schemas.training import (
    TrainingRequest, 
    TrainingResponse, 
    TrainingProgress, 
    ErrorResponse
)
from scripts.train_all import train as train_function
from common.pipeline import get_active_pipelines, stop_pipeline
from services.training_service import (
    start_job,
    cancel_job,
    get_job,
    list_jobs,
    list_active_pipelines as service_active_pipelines,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def convert_request_to_kwargs(request: TrainingRequest) -> Dict[str, Any]:
    """Convert the API request to kwargs for the train function."""
    kwargs = {
        "tenant_id": request.tenant_id,
        "verbose": request.verbose,
        "save_intermediate": True,  # Always save intermediate results for API requests
        "custom_name": request.custom_name,
        "resume_from_checkpoint": request.resume_from_checkpoint,
        "wandb_tags": request.wandb_tags or [],
    }
    
    # Convert Pydantic models to dictionaries, excluding None values
    if request.training_params:
        training_dict = request.training_params.model_dump(exclude_none=True)
        kwargs["training_kwargs"] = training_dict

    if request.model_params:
        model_dict = request.model_params.model_dump(exclude_none=True)
        kwargs["model_kwargs"] = model_dict

    return kwargs


# Background execution and job tracking handled by services/training_service


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
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Convert request to kwargs for the train function
        kwargs = convert_request_to_kwargs(request)
        
        # Delegate job start to the training service (runs in background)
        scheduled_task_id = await start_job(kwargs)

        logger.info(f"Started training task {scheduled_task_id} for tenant {request.tenant_id}")

        job = get_job(scheduled_task_id)
        return TrainingResponse(
            status="accepted",
            task_id=scheduled_task_id,
            message=f"Training job started",
            run_guid=job.get("run_guid") if job else None
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
        message=f"Training job {task_id} is {job['status']}",
        run_guid=job.get("run_guid")
    )


@router.get("/train/jobs", response_model=Dict[str, Any])
async def list_training_jobs():
    """
    List all training jobs and their statuses.
    """
    jobs = list_jobs()
    jobs_summary = {}
    for tid, job in jobs.items():
        jobs_summary[tid] = {
            "status": job.get("status"),
            "tenant_id": job.get("request", {}).get("tenant_id"),
            "custom_name": job.get("request", {}).get("custom_name"),
            "created_at": job.get("created_at"),
            "progress_status": job.get("progress", {}).get("status"),
            "progress_message": job.get("progress", {}).get("message"),
            "run_guid": job.get("run_guid")
        }

    return {
        "total_jobs": len(jobs),
        "jobs": jobs_summary
    }


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
