"""Training endpoints for the LaxAI service."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from schemas.training import TrainingRequest, TrainingResponse, TrainingStatus
from workflows.training_workflow import TrainingWorkflow

router = APIRouter(prefix="/train", tags=["training"])

# In-memory storage for training tasks (in production, use a database)
training_tasks: Dict[str, Dict[str, Any]] = {}


def execute_training_task(task_id: str, training_request: TrainingRequest):
    """Execute actual training using the new TrainingWorkflow."""
    try:
        # Update task status to running
        training_tasks[task_id].update({
            "status": "running",
            "progress": 0,
            "current_epoch": 0,
            "updated_at": datetime.utcnow().isoformat()
        })

        # Extract parameters from request
        training_kwargs = training_request.training_params or {}
        model_kwargs = training_request.model_params or {}
        eval_kwargs = training_request.eval_params or {}

        # Create and execute training workflow
        workflow = TrainingWorkflow(
            tenant_id=getattr(training_request, 'tenant_id', 'tenant1'),
            verbose=getattr(training_request, 'verbose', True),
            save_intermediate=getattr(training_request, 'save_intermediate', True),
            custom_name=training_request.experiment_name or f"api_training_{task_id}",
            resume_from_checkpoint=getattr(training_request, 'resume_from_checkpoint', True),
            wandb_tags=getattr(training_request, 'wandb_tags', []),
            training_kwargs=training_kwargs,
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            pipeline_name=f"api_{task_id}",
            n_datasets_to_use=getattr(training_request, 'n_datasets_to_use', None)
        )

        # Execute the workflow
        result = workflow.execute()

        # Update task with final results
        training_tasks[task_id].update({
            "status": "completed" if result["successful_runs"] > 0 else "failed",
            "progress": 100,
            "datasets_found": result["datasets_found"],
            "successful_runs": result["successful_runs"],
            "total_runs": result["total_runs"],
            "training_results": result["training_results"],
            "updated_at": datetime.utcnow().isoformat()
        })

    except Exception as e:
        # Mark as failed
        training_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.utcnow().isoformat()
        })


@router.post("/", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """Start a new training job."""
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Store task information
    training_tasks[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "experiment_name": request.experiment_name,
        "description": request.description,
        "training_params": request.training_params,
        "model_params": request.model_params,
        "eval_params": request.eval_params,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": None,
        "current_epoch": None,
        "total_epochs": None,
        "loss": None,
        "metrics": None,
        "logs": []
    }
    
    # Start training task in background
    background_tasks.add_task(execute_training_task, task_id, request)
    
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
    
    if task["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel training job in '{task['status']}' status"
        )
    
    # Mark as cancelled
    training_tasks[task_id].update({
        "status": "cancelled",
        "updated_at": datetime.utcnow().isoformat()
    })
    
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
                "experiment_name": task["experiment_name"],
                "status": task["status"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"]
            }
            for task_id, task in training_tasks.items()
        ],
        "total": len(training_tasks)
    }
