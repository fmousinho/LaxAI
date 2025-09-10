"""Service layer for training job lifecycle management.

This module centralizes job creation, tracking, starting (background execution),
and cancellation. The API endpoints should delegate to functions in this
module so the endpoint code stays thin and the business logic is testable.
"""
import asyncio
import logging
import time
import traceback
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import BackgroundTasks

from shared_libs.utils.env_secrets import setup_environment_secrets

# Try to setup environment secrets, but don't fail if unavailable
try:
    setup_environment_secrets()
except (ValueError, ImportError) as e:
    # In cloud environments, secrets might not be available during import
    # They should be loaded when actually needed
    import logging
    logging.getLogger(__name__).warning(f"Could not setup environment secrets during import: {e}")

# Import training dependencies only when needed to avoid heavy ML deps in cloud functions
# from scripts.train_all import train as train_function
# from common.pipeline import get_active_pipelines, stop_pipeline

logger = logging.getLogger(__name__)


def validate_training_params(payload) -> None:
	"""
	Validate training_params against the dynamic TrainingConfig.

	Raises a pydantic.ValidationError (or other exception) on invalid input.
	This function imports the schema at call time to avoid import cycles
	during module initialization.
	"""
	try:
		# Import the dynamic TrainingConfig here to avoid circular imports
		from v1.schemas.training import TrainingConfig
	except Exception as e:
		# If the dynamic schema cannot be imported, raise a clear error
		raise RuntimeError(f"Failed to load TrainingConfig for validation: {e}")

	# Normalize payload
	if payload is None:
		payload_to_validate = {}
	else:
		if hasattr(payload, 'model_dump'):
			payload_to_validate = payload.model_dump()
		else:
			payload_to_validate = payload if isinstance(payload, dict) else dict(payload)

	# Use pydantic model validate; will raise on errors
	TrainingConfig.model_validate(payload_to_validate)


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
		"n_datasets_to_use": getattr(request, 'n_datasets_to_use', None),
	}

	# If n_datasets_to_use is explicitly set to 0, treat it as None (use all datasets)
	if kwargs["n_datasets_to_use"] == 0:
		kwargs["n_datasets_to_use"] = None



	# Training and model params may be Pydantic models
	# training_params and model_params may be provided as Pydantic models,
	# dicts (from parsed JSON), or raw JSON strings from some clients. Be
	# defensive and coerce into dicts when possible.
	tp = getattr(request, 'training_params', None)
	if tp is not None:
		if hasattr(tp, 'model_dump'):
			training_dict = tp.model_dump(exclude_none=True)
		elif isinstance(tp, dict):
			training_dict = tp
		elif isinstance(tp, str):
			try:
				import json
				training_dict = json.loads(tp)
			except Exception:
				training_dict = {"raw": tp}
		else:
			training_dict = {"value": tp}
		kwargs["training_kwargs"] = training_dict
		logger.info(f"TrainingService: Converted training_params to training_kwargs: {training_dict}")

	mp = getattr(request, 'model_params', None)
	if mp is not None:
		if hasattr(mp, 'model_dump'):
			model_dict = mp.model_dump(exclude_none=True)
		elif isinstance(mp, dict):
			model_dict = mp
		elif isinstance(mp, str):
			try:
				import json
				model_dict = json.loads(mp)
			except Exception:
				model_dict = {"raw": mp}
		else:
			model_dict = {"value": mp}
		kwargs["model_kwargs"] = model_dict

	ep = getattr(request, 'eval_params', None)
	if ep is not None:
		if hasattr(ep, 'model_dump'):
			eval_dict = ep.model_dump(exclude_none=True)
		elif isinstance(ep, dict):
			eval_dict = ep
		elif isinstance(ep, str):
			try:
				import json
				eval_dict = json.loads(ep)
			except Exception:
				eval_dict = {"raw": ep}
		else:
			eval_dict = {"value": ep}
		kwargs["eval_kwargs"] = eval_dict

	# If callers accidentally included `n_datasets_to_use` inside the dynamic
	# training kwargs, promote it to the top-level `n_datasets_to_use` unless a
	# top-level value was explicitly provided. This prevents conflicting
	# values and keeps dataset selection deterministic

	return kwargs


async def _run_training_task(task_id: str, kwargs: Dict[str, Any]):
	"""Background runner that executes the train function and updates job state.

	This is intentionally similar to the previous implementation inside the API
	endpoint but lives in the service where it can be unit-tested.
	"""
	try:
		# Import training function here to avoid heavy ML dependencies during module import
		from service_training.src.train_all import train as train_function
		
		TRAINING_JOBS[task_id]["status"] = "running"
		TRAINING_JOBS[task_id]["progress"]["status"] = "initializing"
		TRAINING_JOBS[task_id]["progress"]["message"] = "Starting training pipeline..."

		logger.info(f"Starting training task {task_id} with kwargs: {kwargs}")
		
		# Log training_kwargs specifically for batch_size tracking
		if 'training_kwargs' in kwargs and 'batch_size' in kwargs['training_kwargs']:
			logger.info(f"TrainingService: Starting training with batch_size={kwargs['training_kwargs']['batch_size']}")

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

		# Pipeline name is already set correctly when job was created
		# No need to update it from result

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
		logger.info(f"TrainingService: Training task {task_id} was cancelled: {cancel_msg}")
		logger.debug(f"TrainingService: Cancellation occurred during training execution for task {task_id}")

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


def create_job(request) -> Tuple[str, Dict[str, Any]]:
	"""Create a job entry and return (task_id, kwargs) to start it.

	The caller should call `start_job(task_id, kwargs, background_tasks)` to schedule execution.
	"""
	task_id = str(uuid.uuid4())

	# Convert request to kwargs for training function
	kwargs = _convert_request_to_kwargs(request)

	# Always use the task_id as the canonical pipeline name so cancellation
	# requests can reliably reference the running pipeline by job id.
	# Always use the task_id as the canonical pipeline name so cancellation
	# requests can reliably reference the running pipeline by job id.
	pipeline_name = task_id
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
	logger.info(f"CancellationService: Starting cancellation process for task {task_id}")

	job = TRAINING_JOBS.get(task_id)
	if not job:
		logger.warning(f"CancellationService: Job {task_id} not found in tracking store")
		return False

	current_status = job.get("status", "unknown")
	logger.info(f"CancellationService: Job {task_id} current status: {current_status}")

	if current_status not in ["pending", "running"]:
		logger.info(f"CancellationService: Job {task_id} not in cancellable state (status: {current_status})")
		return False

	pipeline_name = job.get("pipeline_name")
	logger.info(f"CancellationService: Job {task_id} pipeline name: {pipeline_name}")

	if pipeline_name:
		logger.info(f"CancellationService: Attempting to stop pipeline {pipeline_name} for task {task_id}")

		# Import here to avoid heavy ML dependencies during module import
		from common.pipeline import stop_pipeline

		stopped = stop_pipeline(pipeline_name)
		if stopped:
			job["status"] = "cancelling"
			job["progress"]["status"] = "cancelling"
			job["progress"]["message"] = "Training pipeline cancellation requested"
			logger.info(f"CancellationService: Successfully requested cancellation for pipeline {pipeline_name} (task {task_id})")
			logger.debug(f"CancellationService: Pipeline {pipeline_name} stop request sent - waiting for graceful shutdown")
			return True
		else:
			job["status"] = "cancelled"
			job["progress"]["status"] = "cancelled"
			job["progress"]["message"] = "Training job marked as cancelled (pipeline not found or already stopped)"
			logger.warning(f"CancellationService: Pipeline {pipeline_name} not found or already stopped; marked job {task_id} as cancelled in tracking only")
			logger.debug(f"CancellationService: Pipeline registry lookup failed for {pipeline_name} - job marked cancelled locally")
			return False
	else:
		# Pipeline name missing; mark cancelled in tracker
		job["status"] = "cancelled"
		job["progress"]["status"] = "cancelled"
		job["progress"]["message"] = "Training job marked as cancelled (pipeline not yet registered)"
		logger.warning(f"CancellationService: Pipeline name not set for job {task_id}; marked as cancelled in tracker only")
		logger.debug(f"CancellationService: Job {task_id} created without pipeline_name - likely cancelled before pipeline registration")
		return False