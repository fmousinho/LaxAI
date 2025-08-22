"""Service layer for training job lifecycle management.

This module centralizes job creation, tracking, starting (background execution),
and cancellation. The API endpoints should delegate to functions in this
module so the endpoint code stays thin and the business logic is testable.
"""
import asyncio
import logging
import traceback
import uuid
import time
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks

from scripts.train_all import train as train_function
from common.pipeline import get_active_pipelines, stop_pipeline

logger = logging.getLogger(__name__)

# In-memory job store. In production, replace with persistent store (DB/Redis).
TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}


def _convert_request_to_kwargs(request) -> Dict[str, Any]:
	"""Convert an incoming API request model to kwargs for train_function.

	Accepts either a Pydantic model with .model_dump() or a dict-like object.
	"""
	kwargs: Dict[str, Any] = {
		"tenant_id": request.tenant_id,
		"verbose": getattr(request, 'verbose', True),
		"save_intermediate": True,
		"custom_name": getattr(request, 'custom_name', None),
		"resume_from_checkpoint": getattr(request, 'resume_from_checkpoint', True),
		"wandb_tags": getattr(request, 'wandb_tags', []) or [],
		"n_datasets_to_process": getattr(request, 'n_datasets_to_process', None),
	}

	# Training and model params may be Pydantic models
	if getattr(request, 'training_params', None):
		training_dict = request.training_params.model_dump(exclude_none=True)
		kwargs["training_kwargs"] = training_dict

	if getattr(request, 'model_params', None):
		model_dict = request.model_params.model_dump(exclude_none=True)
		kwargs["model_kwargs"] = model_dict

	return kwargs


async def _run_training_task(task_id: str, kwargs: Dict[str, Any]):
	"""Background runner that executes the train function and updates job state.

	This is intentionally similar to the previous implementation inside the API
	endpoint but lives in the service where it can be unit-tested.
	"""
	try:
		TRAINING_JOBS[task_id]["status"] = "running"
		TRAINING_JOBS[task_id]["progress"]["status"] = "initializing"
		TRAINING_JOBS[task_id]["progress"]["message"] = "Starting training pipeline..."

		logger.info(f"Starting training task {task_id} with kwargs: {kwargs}")

		TRAINING_JOBS[task_id]["progress"]["status"] = "running"
		TRAINING_JOBS[task_id]["progress"]["message"] = "Training in progress..."

		loop = asyncio.get_event_loop()

		def training_wrapper():
			try:
				return train_function(**kwargs)
			except Exception:
				# Let exception propagate to outer context
				raise

		result = await loop.run_in_executor(None, training_wrapper)

		if isinstance(result, dict) and result.get("pipeline_name"):
			TRAINING_JOBS[task_id]["pipeline_name"] = result.get("pipeline_name")

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
		cancel_msg = f"Training cancelled: {str(e)}"
		logger.info(f"Training task {task_id} was cancelled: {cancel_msg}")

		TRAINING_JOBS[task_id]["status"] = "cancelled"
		TRAINING_JOBS[task_id]["progress"]["status"] = "cancelled"
		TRAINING_JOBS[task_id]["progress"]["message"] = cancel_msg

	except Exception as e:
		error_msg = f"Training failed: {str(e)}"
		error_details = traceback.format_exc()
		logger.error(f"Training task {task_id} failed: {error_msg}")
		logger.error(f"Error details: {error_details}")

		TRAINING_JOBS[task_id]["status"] = "failed"
		TRAINING_JOBS[task_id]["progress"]["status"] = "failed"
		TRAINING_JOBS[task_id]["progress"]["message"] = error_msg
		TRAINING_JOBS[task_id]["error"] = {"message": error_msg, "details": error_details}


def create_job(request, pipeline_name: Optional[str] = None) -> (str, Dict[str, Any]):
	"""Create a job entry and return (task_id, kwargs) to start it.

	The caller should call `start_job(task_id, kwargs, background_tasks)` to schedule execution.
	"""
	task_id = str(uuid.uuid4())

	# Convert request to kwargs for training function
	kwargs = _convert_request_to_kwargs(request)

	# If caller didn't provide pipeline_name, generate one now
	pipeline_name = pipeline_name or f"training_pipeline_{task_id}"
	kwargs["pipeline_name"] = pipeline_name

	TRAINING_JOBS[task_id] = {
		"status": "pending",
		"request": request.model_dump() if hasattr(request, 'model_dump') else dict(request),
		"WandB_name": getattr(request, 'custom_name', None),
		"pipeline_name": pipeline_name,
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
		"created_at": time.time()
	}

	return task_id, kwargs


def start_job(task_id: str, kwargs: Dict[str, Any], background_tasks: BackgroundTasks) -> None:
	"""Schedule the background training task for execution."""
	# Schedule the async runner
	background_tasks.add_task(_run_training_task, task_id, kwargs)


def get_job(task_id: str) -> Optional[Dict[str, Any]]:
	return TRAINING_JOBS.get(task_id)


def list_jobs() -> Dict[str, Dict[str, Any]]:
	return TRAINING_JOBS.copy()


def cancel_job(task_id: str) -> bool:
	"""Attempt to cancel a running/pending job.

	Returns True if a pipeline stop was requested; False if job was only marked cancelled
	because pipeline was not registered/found.
	"""
	job = TRAINING_JOBS.get(task_id)
	if not job:
		return False

	if job["status"] not in ["pending", "running"]:
		# Nothing to do
		return False

	pipeline_name = job.get("pipeline_name")
	if pipeline_name:
		stopped = stop_pipeline(pipeline_name)
		if stopped:
			job["status"] = "cancelling"
			job["progress"]["status"] = "cancelling"
			job["progress"]["message"] = "Training pipeline cancellation requested"
			logger.info(f"Requested cancellation for pipeline {pipeline_name} (task {task_id})")
			return True
		else:
			job["status"] = "cancelled"
			job["progress"]["status"] = "cancelled"
			job["progress"]["message"] = "Training job marked as cancelled (pipeline not found or already stopped)"
			logger.warning(f"Pipeline {pipeline_name} not found; marked job {task_id} as cancelled in tracking only")
			return False
	else:
		# Pipeline name missing; mark cancelled in tracker
		job["status"] = "cancelled"
		job["progress"]["status"] = "cancelled"
		job["progress"]["message"] = "Training job marked as cancelled (pipeline not yet registered)"
		logger.warning(f"Pipeline name not set for job {task_id}; marked as cancelled in tracker only")
		return False