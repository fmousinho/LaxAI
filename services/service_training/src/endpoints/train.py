"""Training API endpoints for LaxAI service."""

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from services.service_training.src.schemas.training import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
)
from services.service_training.src.schemas.firestore import TrainingJobDocument
from services.service_training.src.state_machine import TrainingJobState, StateTransition
from services.service_training.src.state_manager import TrainingJobStateManager
from services.service_training.src.workflows.training_workflow import TrainingWorkflow

from shared_libs.common.pipeline import stop_pipeline, stop_pipeline_by_guid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])

# Initialize state manager (Firestore as single source of truth)
state_manager = TrainingJobStateManager()


def execute_training_task(task_id: str, training_request: TrainingRequest):
    """Execute actual training using the TrainingWorkflow."""
    try:
        # Update task status to running
        state_manager.update_status(task_id, TrainingJobState.RUNNING, progress=0)

        # Extract parameters from request (already Pydantic objects or None)
        training_params = training_request.training_params
        model_params = training_request.model_params
        eval_params = training_request.eval_params
        
        # Extract dataset_address from training_params if available
        dataset_address = training_params.dataset_address if training_params else None
        logger.info(f"📍 API received dataset_address: {dataset_address}")

        # Create and execute training workflow
        workflow = TrainingWorkflow(
            tenant_id=training_request.tenant_id,
            wandb_run_name=training_request.wandb_run_name,
            training_params=training_params,
            eval_params=eval_params,
            dataset_address=dataset_address,
            task_id=task_id
        )

        # Execute the workflow
        result = workflow.execute()

        # Map workflow status to TrainingJobState
        workflow_status = result.get("status", "completed")
        if workflow_status == "completed":
            final_state = TrainingJobState.COMPLETED
        elif workflow_status == "cancelled":
            final_state = TrainingJobState.CANCELLED
        elif workflow_status == "auto_suspended":
            final_state = TrainingJobState.AUTO_SUSPENDED
        else:
            final_state = TrainingJobState.FAILED

        # Check if task was externally cancelled
        current_job = state_manager.get_job(task_id)
        if current_job and current_job.status == TrainingJobState.CANCELLED:
            return

        # Update final status with results
        state_manager.update_status(
            task_id,
            final_state,
            progress=100 if final_state == TrainingJobState.COMPLETED else None,
            run_guids=result.get("run_guids", []),
        )

    except InterruptedError:
        # Handle cancellation
        state_manager.update_status(task_id, TrainingJobState.CANCELLED)
    except Exception as e:
        # Mark as failed
        state_manager.update_status(task_id, TrainingJobState.FAILED, error=str(e))
        logger.exception(f"Training task {task_id} failed")


@router.post("/", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks  # Kept for signature compatibility
) -> TrainingResponse:
    """Start a new training job."""

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
    state_manager.create_job(job)

    # Start training in background thread
    threading.Thread(target=execute_training_task, args=(task_id, request), daemon=True).start()

    return TrainingResponse(
        task_id=task_id,
        status=TrainingJobState.QUEUED.value,
        message="Training job has been queued successfully",
        created_at=job.created_at.isoformat()
    )


@router.get("/{task_id}", response_model=TrainingStatus)
async def get_training_status(task_id: str) -> TrainingStatus:
    """Get the status of a training job."""

    job = state_manager.get_job(task_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

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
async def cancel_training(task_id: str) -> JSONResponse:
    """Cancel a training job."""

    job = state_manager.get_job(task_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Check if already in terminal state
    if StateTransition.is_terminal_state(job.status):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel training job in '{job.status.value}' status"
        )

    # Update status to cancelled (with validation)
    success = state_manager.update_status(task_id, TrainingJobState.CANCELLED)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state transition from '{job.status.value}' to 'cancelled'"
        )

    # Stop active pipelines using GUIDs
    cancelled_count = 0
    for run_guid in job.run_guids:
        if stop_pipeline_by_guid(run_guid):
            cancelled_count += 1
    
    logger.info(f"Cancelled {cancelled_count} pipeline(s) for task {task_id}")

    return JSONResponse(
        content={
            "message": "Training job cancelled successfully",
            "task_id": task_id,
            "status": "cancelled"
        }
    )


@router.get("/")
async def list_training_jobs() -> Dict[str, Any]:
    """List all training jobs (from Firestore)."""
    
    # Note: For production, you'd want to add pagination and filtering
    # For now, this is a simple implementation
    # In a real implementation, you'd query Firestore collection
    
    return {
        "training_jobs": [],  # Would query Firestore collection
        "total": 0,
        "message": "List endpoint requires Firestore collection query implementation"
    }
