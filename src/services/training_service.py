"""
Training service that provides a thin orchestration layer between the API/CLI
and the pipeline execution code.

Responsibilities:
- Maintain an in-memory job registry mapping task_id -> job metadata
- Start training jobs in the background using the same train function used by
  the CLI (`scripts.train_all.train`)
- Capture pipeline `run_guid` from pipeline results and store the mapping
- Provide cancel, list and get operations that use the pipeline control API
"""
import asyncio
import logging
import uuid
import traceback
from typing import Dict, Any, Optional
from concurrent.futures import Future

from scripts.train_all import train as train_function
from common.pipeline import stop_pipeline, get_active_pipelines

logger = logging.getLogger(__name__)

# Internal job registry keyed by run_guid (set once pipeline starts)
TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}

# Pending jobs keyed by task_id until run_guid is available
PENDING_JOBS: Dict[str, Dict[str, Any]] = {}

# Map task_id -> run_guid when available
TASK_TO_RUN: Dict[str, str] = {}

# Cancellation flags for long-running tasks (optional usage)
CANCELLATION_FLAGS: Dict[str, bool] = {}


# Module-level helpers -----------------------------------------------------
def _get_job_record_for_task_global(tid: str) -> Optional[Dict[str, Any]]:
    run = TASK_TO_RUN.get(tid)
    if run:
        return TRAINING_JOBS.get(run)
    return PENDING_JOBS.get(tid)


def _attach_run_guid_global(tid: str, run_guid: str):
    """Move a pending job indexed by task_id into TRAINING_JOBS keyed by run_guid."""
    if TASK_TO_RUN.get(tid):
        return
    pending = PENDING_JOBS.pop(tid, None)
    if pending is None:
        pending = {
            "task_id": tid,
            "run_guid": run_guid,
            "status": "running",
            "progress": {"status": "running", "message": "Training started"},
            "future": None,
            "created_at": asyncio.get_event_loop().time(),
            "error": None,
        }
    pending["run_guid"] = run_guid
    TRAINING_JOBS[run_guid] = pending
    TASK_TO_RUN[tid] = run_guid



async def start_job(kwargs: Dict[str, Any]) -> str:
    """Start a training job in the background and return the generated task_id.

    This function schedules the blocking `train_function` to run in a thread
    pool executor and returns the task id immediately.
    """
    task_id = str(uuid.uuid4())

    # Ensure pipeline_name is set so pipeline will be registered with an
    # identifiable name (helps for human-readable logs). We still prefer
    # run_guid for identification but keeping pipeline_name is useful.
    kwargs = dict(kwargs)

    # Initialize pending job record (we don't yet have run_guid)
    PENDING_JOBS[task_id] = {
        "task_id": task_id,
        "run_guid": None,
        "status": "pending",
        "progress": {
            "status": "pending",
            "message": "Job queued"
        },
        "future": None,
        "created_at": asyncio.get_event_loop().time(),
        "error": None,
    }

    loop = asyncio.get_running_loop()

    # Attach an on_started callback so we capture run_guid as soon as the
    # TrainPipeline is instantiated (it sets run_guid in its constructor).
    def _on_started(run_guid: str):
        # Move pending -> active as soon as run_guid is available
        _attach_run_guid_global(task_id, run_guid)

    # Inject the callback into kwargs so the shared train() can call it early
    kwargs_with_callback = dict(kwargs)
    kwargs_with_callback["on_started"] = _on_started

    # Schedule the blocking work in default executor
    future = loop.run_in_executor(None, _run_job_sync, task_id, kwargs_with_callback)

    # Store the future and update status in the pending record
    PENDING_JOBS[task_id]["future"] = future
    PENDING_JOBS[task_id]["status"] = "running"
    PENDING_JOBS[task_id]["progress"]["status"] = "running"
    PENDING_JOBS[task_id]["progress"]["message"] = "Training started"
    # Keep a copy of the original request kwargs for introspection
    PENDING_JOBS[task_id]["request"] = dict(kwargs)

    # Give the background thread a short window to invoke the on_started
    # callback so we can capture run_guid synchronously for the API caller.
    async def _wait_for_run_guid(timeout: float = 1.0) -> Optional[str]:
        elapsed = 0.0
        interval = 0.05
        while elapsed < timeout:
            rg = PENDING_JOBS.get(task_id, {}).get("run_guid")
            if rg:
                return rg
            await asyncio.sleep(interval)
            elapsed += interval
        return None

    try:
        # Wait up to 1 second for run_guid to appear from the on_started callback
        run_guid = await asyncio.wait_for(_wait_for_run_guid(), timeout=1.0)
        if run_guid:
            # Move pending -> active record
            _attach_run_guid_global(task_id, run_guid)
    except asyncio.TimeoutError:
        # Not critical; the run_guid will be set shortly by the background thread
        pass

    # Attach a callback to update job metadata when done
    def _on_done(fut: Future):
        job = _get_job_record_for_task_global(task_id)
        try:
            result = fut.result()
            # result may be the pipeline results dict (if train_function returns it)
            if isinstance(result, dict):
                run_guid = result.get("run_guid")
                if run_guid:
                    _attach_run_guid_global(task_id, run_guid)
                    job = _get_job_record_for_task_global(task_id)
                # mark completed if not already set
                if job is not None:
                    job["status"] = result.get("status", "completed")
                    job["progress"]["status"] = job["status"]
                    job["progress"]["message"] = "Training finished"
            else:
                # If no structured result, just mark completed
                if job is not None:
                    job["status"] = "completed"
                    job["progress"]["status"] = "completed"
                    job["progress"]["message"] = "Training finished"
        except InterruptedError as e:
            if job is not None:
                job["status"] = "cancelled"
                job["progress"]["status"] = "cancelled"
                job["progress"]["message"] = f"Cancelled: {e}"
        except Exception as e:
            if job is not None:
                job["status"] = "failed"
                job["progress"]["status"] = "failed"
                job["progress"]["message"] = str(e)
                job["error"] = {
                    "message": str(e),
                    "details": traceback.format_exc()
                }
        finally:
            # Ensure future cleared (no strong refs)
            if job is not None:
                job["future"] = None

    future.add_done_callback(_on_done)

    logger.info(f"Scheduled training job {task_id}")
    return task_id


def _run_job_sync(task_id: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Synchronous runner executed in a thread pool. Calls the project's
    `train` function (used by CLI) and captures its output.

    Returns whatever the train function returns (often a dict with run_guid).
    """
    CANCELLATION_FLAGS[task_id] = False
    logger.info(f"Running training job {task_id} (sync) with kwargs: {kwargs}")

    try:
        # Call the shared train function (blocking). The train() function accepts
        # an optional on_started callback which is called immediately after the
        # TrainPipeline instance is created and its run_guid is available.
        # Filter kwargs to only parameters accepted by the train function to
        # avoid passing internal keys like 'pipeline_name'.
        import inspect
        sig = inspect.signature(train_function)
        accepted_params = set(sig.parameters.keys())
        call_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
        removed = [k for k in kwargs.keys() if k not in accepted_params]
        if removed:
            logger.debug(f"Filtered out unexpected kwargs before calling train: {removed}")

        result = train_function(**call_kwargs)

        # Try to capture run_guid from result
        if isinstance(result, dict):
            run_guid = result.get("run_guid")
            if run_guid:
                # If train() returned a run_guid, attach it to the pending job
                _attach_run_guid_global(task_id, run_guid)

        logger.info(f"Training job {task_id} completed successfully")
        return result

    except InterruptedError:
        logger.info(f"Training job {task_id} was interrupted/cancelled")
        raise
    except Exception:
        logger.error(f"Training job {task_id} failed: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup the cancellation flag
        CANCELLATION_FLAGS.pop(task_id, None)


def cancel_job(task_id: str) -> bool:
    """Request cancellation for a job by task_id.

    Returns True if a pipeline stop was requested, False otherwise.
    """
    # Determine whether caller provided a run_guid directly or a task_id.
    # Priority:
    # 1) If the identifier matches a run_guid in TRAINING_JOBS, treat it as run_guid.
    # 2) Else, if TASK_TO_RUN maps the task_id to a run_guid, use that.
    # 3) Else, if the job is still pending, mark it cancelled locally.
    job = None
    pipeline_identifier = None

    if task_id in TRAINING_JOBS:
        # Caller passed a run_guid directly
        pipeline_identifier = task_id
        job = TRAINING_JOBS.get(pipeline_identifier)
    else:
        # Try to resolve task_id -> run_guid
        run_guid = TASK_TO_RUN.get(task_id)
        if run_guid:
            pipeline_identifier = run_guid
            job = TRAINING_JOBS.get(run_guid)
        elif task_id in PENDING_JOBS:
            # Still pending, mark as cancelled locally
            job = PENDING_JOBS.pop(task_id)
            job["status"] = "cancelled"
            job["progress"]["status"] = "cancelled"
            job["progress"]["message"] = "Marked as cancelled (pending)"
            CANCELLATION_FLAGS[task_id] = True
            logger.warning(f"Pending job {task_id} cancelled locally before pipeline start")
            return False
        else:
            return False

    stopped = stop_pipeline(pipeline_identifier)

    if stopped:
        if job is not None:
            job["status"] = "cancelling"
            job["progress"]["status"] = "cancelling"
            job["progress"]["message"] = "Cancellation requested"
        logger.info(f"Requested stop for pipeline {pipeline_identifier} (task {task_id})")
    else:
        # Mark cancelled in job tracking if pipeline not found
        if job is not None:
            job["status"] = "cancelled"
            job["progress"]["status"] = "cancelled"
            job["progress"]["message"] = "Marked as cancelled (pipeline not found)"
        logger.warning(f"Pipeline {pipeline_identifier} not found for task {task_id}; marked cancelled locally")

    # Also set cancellation flag so in-process code that checks it can respond
    CANCELLATION_FLAGS[task_id] = True

    return bool(stopped)


def get_job(task_id: str) -> Optional[Dict[str, Any]]:
    """Resolve and return a job record by task_id or run_guid.

    If a task_id was provided and mapped to a run_guid, return the active
    record keyed by run_guid. Otherwise, return a pending record if present.
    """
    # If caller passed a run_guid directly
    if task_id in TRAINING_JOBS:
        return TRAINING_JOBS.get(task_id)

    # If we have a mapping from task_id -> run_guid
    run = TASK_TO_RUN.get(task_id)
    if run:
        return TRAINING_JOBS.get(run)

    # Fallback to pending jobs keyed by task_id
    return PENDING_JOBS.get(task_id)


def list_jobs() -> Dict[str, Dict[str, Any]]:
    """Return a combined view of pending and active jobs.

    Active jobs are keyed by run_guid. Pending jobs remain keyed by task_id.
    """
    combined: Dict[str, Dict[str, Any]] = {}
    # Active jobs (run_guid keys)
    combined.update(TRAINING_JOBS)
    # Pending jobs (task_id keys) - prefix to avoid key collision
    for tid, rec in PENDING_JOBS.items():
        combined[f"pending:{tid}"] = rec
    return combined


def list_active_pipelines() -> Dict[str, str]:
    return get_active_pipelines()
