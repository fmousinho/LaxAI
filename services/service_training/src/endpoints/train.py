"""Training API endpoints for LaxAI service."""

import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from services.service_training.src.schemas.training import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
)
from services.service_training.src.workflows.training_workflow import TrainingWorkflow

from shared_libs.common.pipeline import stop_pipeline, stop_pipeline_by_guid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])

# In-memory storage for training tasks (in production, use a database)
training_tasks: Dict[str, Dict[str, Any]] = {}


def execute_training_task(task_id: str, training_request: TrainingRequest):
    """Execute actual training using the TrainingWorkflow."""
    try:
        # Update task status to running
        training_tasks[task_id].update({
            "status": "running",
            "progress": 0,
            "current_epoch": 0,
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

        # Extract parameters from request (already Pydantic objects or None)
        training_params = training_request.training_params
        model_params = training_request.model_params
        eval_params = training_request.eval_params
        
        # Extract n_datasets_to_use and dataset_address from training_params if available
        n_datasets_to_use = training_params.n_datasets_to_use if training_params else None
        dataset_address = training_params.dataset_address if training_params else None
        logger.info(f"🎯 API received n_datasets_to_use: {n_datasets_to_use} (type: {type(n_datasets_to_use)})")
        logger.info(f"📍 API received dataset_address: {dataset_address}")

        # Create and execute training workflow
        workflow = TrainingWorkflow(
            tenant_id=training_request.tenant_id,
            verbose=getattr(training_request, 'verbose', True),
            custom_name=training_request.wandb_run_name,
            wandb_tags=getattr(training_request, 'wandb_tags', []),
            training_params=training_params,
            model_params=model_params,
            eval_params=eval_params,
            pipeline_name=f"api_{task_id}",
            n_datasets_to_use=n_datasets_to_use,
            dataset_address=dataset_address,
            task_id=task_id
        )

        # Execute the workflow (single pipeline)
        result = workflow.execute()

        # Map workflow status directly
        final_status = result.get("status", "completed")

        # If the task was externally cancelled (status already set to 'cancelled'),
        # don't overwrite with a later completed/failed status from the workflow.
        current_task_status = training_tasks.get(task_id, {}).get("status")
        if current_task_status == "cancelled":
            return

        training_tasks[task_id].update({
            "status": final_status,
            "progress": 100 if final_status == "completed" else None,
            "datasets_found": result.get("datasets_found", 0),
            "steps_completed": result.get("steps_completed", 0),
            "dataset_mode": result.get("dataset_mode"),
            "run_guids": result.get("run_guids", []),
            "pipeline_result": result.get("pipeline_result"),
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

    except InterruptedError:
        # Handle cancellation
        training_tasks[task_id].update({
            "status": "cancelled",
            "updated_at": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        # Mark as failed
        training_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat()
        })


@router.post("/", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks  # Kept for signature compatibility, not used directly now
) -> TrainingResponse:
    """Start a new training job."""

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Store task information
    training_tasks[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "wandb_run_name": request.wandb_run_name,
        "tenant_id": request.tenant_id,
        "training_params": request.training_params.dict() if request.training_params else {},
        "model_params": request.model_params.dict() if request.model_params else {},
        "eval_params": request.eval_params.dict() if request.eval_params else {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "progress": None,
        "current_epoch": None,
        "total_epochs": None,
        "loss": None,
        "metrics": None,
        "logs": [],
        "pipeline_name": f"api_{task_id}"  # Store pipeline name for cancellation
    }

    # Previous implementation used FastAPI BackgroundTasks. Under Starlette's TestClient
    # the response isn't returned until background tasks finish, which blocked the API
    # call for the entire (long) training duration and prevented issuing a cancel
    # request from the tests. We switch to spawning a real daemon thread so the POST
    # returns immediately and cancellation can be exercised concurrently.
    threading.Thread(target=execute_training_task, args=(task_id, request), daemon=True).start()

    return TrainingResponse(
        task_id=task_id,
        status="queued",
        message="Training job has been queued successfully",
        created_at=training_tasks[task_id]["created_at"]
    )


@router.get("/{task_id}", response_model=TrainingStatus)
async def get_training_status(task_id: str) -> TrainingStatus:
    """Get the status of a training job."""

    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Training job not found")

    task = training_tasks[task_id]

    return TrainingStatus(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        current_epoch=task.get("current_epoch"),
        total_epochs=task.get("total_epochs"),
        loss=task.get("loss"),
        metrics=task.get("metrics"),
        logs=task.get("logs"),
        updated_at=task["updated_at"]
    )


@router.delete("/{task_id}")
async def cancel_training(task_id: str) -> JSONResponse:
    """Cancel a training job."""

    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Training job not found")

    task = training_tasks[task_id]

    if task["status"] == "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel training job in '{task['status']}' status"
        )

    # Mark as cancelled in our task storage
    training_tasks[task_id].update({
        "status": "cancelled",
        "updated_at": datetime.now(timezone.utc).isoformat()
    })

    # Use GUID-based cancellation for all active pipelines
    run_guids = task.get("run_guids", [])
    cancelled_count = 0
    
    for run_guid in run_guids:
        if stop_pipeline_by_guid(run_guid):
            cancelled_count += 1
    
    # Fallback to pipeline name if no GUIDs worked
    if cancelled_count == 0:
        pipeline_name = task.get("pipeline_name", f"api_{task_id}")
        stop_pipeline(pipeline_name)

    return JSONResponse(
        content={
            "message": "Training job cancelled successfully",
            "task_id": task_id,
            "status": "cancelled"
        }
    )


@router.get("/")
async def list_training_jobs() -> Dict[str, Any]:
    """List all training jobs."""

    return {
        "training_jobs": [
            {
                "task_id": task_id,
                "wandb_run_name": task.get("wandb_run_name", ""),
                "tenant_id": task["tenant_id"],
                "status": task["status"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"]
            }
            for task_id, task in training_tasks.items()
        ],
        "total": len(training_tasks)
    }



