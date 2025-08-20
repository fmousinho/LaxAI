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
from scripts.train_all import train as train_function
from common.pipeline import get_active_pipelines, stop_pipeline

logger = logging.getLogger(__name__)

# In-memory storage for tracking training jobs
# In production, you'd want to use a proper database or Redis
TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}

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


async def run_training_task(task_id: str, kwargs: Dict[str, Any]):
    """Background task to run the training process and record pipeline metadata.

    This wrapper captures the return value from `train_function` so we can
    record the actual `pipeline_name` used by the pipeline (produced by
    `scripts.train_all.train`) instead of inventing a name in the API layer.
    """
    try:
        # Update status to running
        TRAINING_JOBS[task_id]["status"] = "running"
        TRAINING_JOBS[task_id]["progress"]["status"] = "initializing"
        TRAINING_JOBS[task_id]["progress"]["message"] = "Starting training pipeline..."

        logger.info(f"Starting training task {task_id} with kwargs: {kwargs}")

        # Update progress
        TRAINING_JOBS[task_id]["progress"]["status"] = "running"
        TRAINING_JOBS[task_id]["progress"]["message"] = "Training in progress..."

        # Run the training function in a thread pool to avoid blocking and
        # capture its return value so we can record the pipeline_name it used.
        loop = asyncio.get_event_loop()

        def training_wrapper():
            try:
                return train_function(**kwargs)
            except Exception:
                # Re-raise in worker thread so outer code sees traceback
                raise

        result = await loop.run_in_executor(None, training_wrapper)

        # If train_function returned a dict with pipeline_name, store it
        if isinstance(result, dict) and result.get("pipeline_name"):
            TRAINING_JOBS[task_id]["pipeline_name"] = result.get("pipeline_name")

        # Update job status using returned status when available
        returned_status = result.get("status") if isinstance(result, dict) else None
        if returned_status == "completed":
            TRAINING_JOBS[task_id]["status"] = "completed"
            TRAINING_JOBS[task_id]["progress"]["status"] = "completed"
            TRAINING_JOBS[task_id]["progress"]["message"] = "Training completed successfully"
            logger.info(f"Training task {task_id} completed successfully")
        else:
            TRAINING_JOBS[task_id]["status"] = returned_status or "finished"
            TRAINING_JOBS[task_id]["progress"]["status"] = returned_status or "finished"
            TRAINING_JOBS[task_id]["progress"]["message"] = "Training finished"
            logger.info(f"Training task {task_id} finished with status: {returned_status}")

    except InterruptedError as e:
        # Handle pipeline cancellation
        cancel_msg = f"Training cancelled: {str(e)}"
        logger.info(f"Training task {task_id} was cancelled: {cancel_msg}")

        # Update status to cancelled
        TRAINING_JOBS[task_id]["status"] = "cancelled"
        TRAINING_JOBS[task_id]["progress"]["status"] = "cancelled"
        TRAINING_JOBS[task_id]["progress"]["message"] = cancel_msg

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        error_details = traceback.format_exc()

        logger.error(f"Training task {task_id} failed: {error_msg}")
        logger.error(f"Error details: {error_details}")

        # Update status to failed
        TRAINING_JOBS[task_id]["status"] = "failed"
        TRAINING_JOBS[task_id]["progress"]["status"] = "failed"
        TRAINING_JOBS[task_id]["progress"]["message"] = error_msg
        TRAINING_JOBS[task_id]["error"] = {
            "message": error_msg,
            "details": error_details
        }


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

        # Initialize job tracking (pipeline_name unknown until train() runs)
        TRAINING_JOBS[task_id] = {
            "status": "pending",
            "request": request.model_dump(),
            "WandB_name": request.custom_name,
            "pipeline_name": None,
            "progress": {
                "status": "pending",
                "message": "Training job queued",
                "current_epoch": None,
                "total_epochs": None,
                "current_loss": None,
                "best_loss": None,
                "datasets_found": None,
                "datasets_processed": None,
                "logs": []
            },
            "created_at": asyncio.get_event_loop().time()
        }

        # Add background task
        background_tasks.add_task(run_training_task, task_id, kwargs)

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
    if task_id not in TRAINING_JOBS:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                detail=f"Training task {task_id} not found",
                error_type="task_not_found"
            ).model_dump()
        )
    
    job = TRAINING_JOBS[task_id]
    
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
    jobs_summary = {}
    for task_id, job in TRAINING_JOBS.items():
        jobs_summary[task_id] = {
            "status": job["status"],
            "tenant_id": job["request"]["tenant_id"],
            "custom_name": job["request"]["custom_name"],
            "created_at": job["created_at"],
            "progress_status": job["progress"]["status"],
            "progress_message": job["progress"]["message"]
        }
    
    return {
        "total_jobs": len(TRAINING_JOBS),
        "jobs": jobs_summary
    }


@router.delete("/train/{task_id}")
async def cancel_training_job(task_id: str):
    """
    Cancel a training job using the pipeline control system.
    """
    if task_id not in TRAINING_JOBS:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                detail=f"Training task {task_id} not found",
                error_type="task_not_found"
            ).model_dump()
        )
    
    job = TRAINING_JOBS[task_id]
    
    if job["status"] in ["pending", "running"]:
        # Use the pipeline_name produced by train_all when available. If the
        # pipeline hasn't registered yet (pipeline_name is None), we cannot
        # stop it via the pipeline registry — mark the job cancelled in the
        # tracker and return. This avoids inventing a name and producing the
        # confusing warning seen previously.
        pipeline_name_to_stop = job.get("pipeline_name")

        if pipeline_name_to_stop:
            pipeline_stopped = stop_pipeline(pipeline_name_to_stop)

            if pipeline_stopped:
                job["status"] = "cancelling"
                job["progress"]["status"] = "cancelling"
                job["progress"]["message"] = "Training pipeline cancellation requested"
                logger.info(f"Successfully requested cancellation of pipeline '{pipeline_name_to_stop}' for task {task_id}")
            else:
                job["status"] = "cancelled"
                job["progress"]["status"] = "cancelled"
                job["progress"]["message"] = "Training job marked as cancelled (pipeline not found or already stopped)"
                logger.warning(f"Pipeline '{pipeline_name_to_stop}' not found for task {task_id}; marked as cancelled in job tracking only")
        else:
            # Pipeline hasn't registered yet; cannot stop via registry without
            # guessing a name. Mark job cancelled in tracking only and inform
            # the user clearly.
            job["status"] = "cancelled"
            job["progress"]["status"] = "cancelled"
            job["progress"]["message"] = "Training job marked as cancelled (pipeline not yet registered)"
            logger.warning(f"Pipeline for task {task_id} has not registered a pipeline_name yet; marked as cancelled in job tracking only")
    
    return {"message": f"Training job {task_id} cancelled", "pipeline_stopped": pipeline_stopped if 'pipeline_stopped' in locals() else False}


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
