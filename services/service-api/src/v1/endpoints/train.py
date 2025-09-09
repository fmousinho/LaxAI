"""
Training API endpoint for LaxAI - Gateway to service-training.
"""
import asyncio
import logging
import os
import traceback
import uuid
from typing import Any, Dict

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..schemas.training import (ErrorResponse, TrainingConfig,
                                TrainingProgress, TrainingRequest,
                                TrainingResponse)

logger = logging.getLogger(__name__)

# The service manages job storage now

router = APIRouter()

# Training service URL (use localhost for development, service name for Docker)
TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://localhost:8001/api/v1")


# Request conversion is handled inside the service


# Background execution handled by service


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a training job by proxying to service-training.
    
    Returns a task ID that can be used to track progress.
    """
    try:
        # Proxy the request to service-training
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TRAINING_SERVICE_URL}/train",
                json=request.model_dump(),
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from service-training: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
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
    Get the status of a training job by proxying to service-training.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TRAINING_SERVICE_URL}/train/{task_id}/status",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    detail=f"Training task {task_id} not found",
                    error_type="task_not_found"
                ).model_dump()
            )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/jobs", response_model=Dict[str, Any])
async def list_training_jobs():
    """
    List all training jobs and their statuses by proxying to service-training.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TRAINING_SERVICE_URL}/train/jobs",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to list training jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/train/{task_id}")
async def cancel_training_job(task_id: str):
    """
    Cancel a training job by proxying to service-training.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{TRAINING_SERVICE_URL}/train/{task_id}",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    detail=f"Training task {task_id} not found",
                    error_type="task_not_found"
                ).model_dump()
            )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Failed to cancel training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/pipelines", response_model=Dict[str, Any])
async def list_active_pipelines():
    """
    List all currently active training pipelines by proxying to service-training.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TRAINING_SERVICE_URL}/train/pipelines",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to list active pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))



