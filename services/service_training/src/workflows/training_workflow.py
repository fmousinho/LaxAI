"""
Training workflow orchestration for LaxAI.

This module contains the core training workflow logic that can be used
by CLI, API, or other interfaces.
"""

import asyncio
import logging
import os
import signal
import threading
from typing import Any, Dict, List, Optional

from schemas.training import TrainingParams, ModelParams, EvalParams

# Enable MPS fallback for unsupported operations, as recommended by PyTorch.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logger = logging.getLogger(__name__)

from shared_libs.common.google_storage import GCSPaths, get_storage  # noqa: E402

try:
    from google.cloud import pubsub_v1
    PUBSUB_AVAILABLE = True
except ImportError:
    PUBSUB_AVAILABLE = False
    logger.warning("google-cloud-pubsub not available, Pub/Sub publishing disabled")
from services.service_training.src.train_pipeline import TrainPipeline  # noqa: E402


class TrainingWorkflow:
    """
    Orchestrates the end-to-end training process for LaxAI.

    This class handles the complete workflow from dataset discovery
    to model training execution.
    """

    def __init__(self,
                 tenant_id: str,
                 wandb_run_name: str = "training_workflow_run",
                 training_params: Optional[TrainingParams] = None,
                 dataset_address: Optional[str] = None,
                 eval_params: Optional[EvalParams] = None,
                 task_id: Optional[str] = None,
                 cancellation_event: Optional[threading.Event] = None,
                 timeout_triggered: Optional[threading.Event] = None,
                 execution_timeout: Optional[int] = None,
                 auto_resume_count: int = 0):
        """
        Initialize the training workflow.

        Args:
            tenant_id: The tenant ID for GCS operations.
            wandb_run_name: Name of the run in Weights & Biases.
            training_params: TrainingParams object with training hyperparameters.
            dataset_address: Full GCS path to a specific dataset (gs://bucket/path or bucket/path).
                           If provided, this overrides dataset discovery and n_datasets_to_use.
            eval_params: EvalParams object with evaluation parameters.
            task_id: Task ID for tracking this training run.
            cancellation_event: Event for graceful cancellation handling.
            timeout_triggered: Event that indicates if cancellation was due to timeout.
            execution_timeout: Maximum execution time in minutes. If set, a message will be sent to pub/sub to suspend and restart the job.
            auto_resume_count: Number of times this job has been auto-resumed.
        """
        self.tenant_id = tenant_id
        self.wandb_run_name = wandb_run_name
        self.training_params = training_params or TrainingParams()
        self.dataset_address = dataset_address
        self.eval_params = eval_params or EvalParams()
        self.task_id = task_id
        self.cancellation_event = cancellation_event
        self.timeout_triggered = timeout_triggered
        self.execution_timeout = execution_timeout
        self.auto_resume_count = auto_resume_count

        # Initialize Firestore client for status updates if task_id is provided
        self.firestore_client = None
        if self.task_id:
            try:
                from google.cloud import firestore
                self.firestore_client = firestore.Client()
                logger.info(f"Initialized Firestore client for task_id: {self.task_id}")
            except ImportError:
                logger.warning("google-cloud-firestore not available, status updates disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Firestore client: {e}")

        # Initialize Pub/Sub publisher client for message publishing
        self.pubsub_publisher = None
        if PUBSUB_AVAILABLE:
            try:
                self.pubsub_publisher = pubsub_v1.PublisherClient()
                logger.info("Initialized Pub/Sub publisher client")
            except Exception as e:
                logger.error(f"Failed to initialize Pub/Sub publisher client: {e}")
                self.pubsub_publisher = None

        # Set up signal handlers for external cancellation (e.g., Cloud Run job cancellation)
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful cancellation."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            # Update Firestore status to cancelled if we have a task_id
            if self.task_id and self.firestore_client:
                try:
                    self._update_firestore_status("cancelled", f"Cancelled by signal {signum}")
                    logger.info(f"Updated Firestore status to cancelled for task_id: {self.task_id}")
                except Exception as e:
                    logger.error(f"Failed to update Firestore status on signal: {e}")
            
            # Set the cancellation event if it exists
            if self.cancellation_event:
                self.cancellation_event.set()
                logger.info("Set cancellation event")
            
            # Stop any active pipelines
            try:
                from shared_libs.common.pipeline import stop_pipeline
                if self.pipeline_name:
                    stop_pipeline(self.pipeline_name)
                    logger.info(f"Requested pipeline stop for: {self.pipeline_name}")
            except Exception as e:
                logger.error(f"Failed to stop pipeline: {e}")

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("Signal handlers registered for graceful cancellation")


    def _update_firestore_status(self, status: str, error: Optional[str] = None) -> None:
        """
        Update the Firestore document with the final training status.

        Args:
            status: Final status (completed, error, cancelled)
            error: Optional error message if status is error
        """
        if not self.firestore_client or not self.task_id:
            return

        try:
            from datetime import datetime, timezone

            doc_ref = self.firestore_client.collection("training_runs").document(self.task_id)

            update_data = {
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat() + "Z"
            }

            if error:
                update_data["error"] = error

            doc_ref.update(update_data)
            logger.info(f"Updated Firestore status for task_id {self.task_id} to: {status}")

        except Exception as e:
            logger.error(f"Failed to update Firestore status for task_id {self.task_id}: {e}")

    def _publish_auto_resume_message(self) -> None:
        """
        Publish an auto-resume message to Pub/Sub with all job parameters.
        This allows the proxy to restart the job with the same configuration.
        """
        if not self.pubsub_publisher:
            logger.error("Cannot publish auto-resume message: Pub/Sub publisher not initialized")
            return

        try:
            from datetime import datetime, timezone
            import json

            topic_path = self.pubsub_publisher.topic_path("laxai-466119", "training-jobs")

            # Ensure training_params has weights set to 'checkpoint' for auto-resume
            training_params_dict = self.training_params.dict() if self.training_params else {}
            training_params_dict['weights'] = 'checkpoint'  # Always resume from checkpoint on auto-resume
            
            # Construct message with all parameters needed to restart the job
            message_data = {
                "action": "auto_resume",
                "tenant_id": self.tenant_id,
                "task_id": self.task_id,
                "training_params": training_params_dict,
                "eval_params": self.eval_params.dict() if self.eval_params else {},
                "dataset_address": self.dataset_address,
                "auto_resume_count": self.auto_resume_count + 1,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            message_json = json.dumps(message_data).encode('utf-8')
            future = self.pubsub_publisher.publish(topic_path, message_json)
            message_id = future.result()
            
            logger.info(f"Published auto-resume message (attempt #{self.auto_resume_count + 1}) for task_id {self.task_id}: {message_id}")

        except Exception as e:
            logger.error(f"Failed to publish auto-resume message for task_id {self.task_id}: {e}")

    def _publish_pubsub_message(self, action: str, task_id: str) -> None:
        """
        Publish a message to the training-jobs Pub/Sub topic.

        Args:
            action: The action to publish (e.g., "suspend-restart")
            task_id: The task ID for the training run
        """
        if not self.pubsub_publisher or not task_id:
            return

        try:
            from datetime import datetime, timezone

            topic_path = self.pubsub_publisher.topic_path("laxai-466119", "training-jobs")

            message_data = {
                "action": action,
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            import json
            message_json = json.dumps(message_data).encode('utf-8')

            future = self.pubsub_publisher.publish(topic_path, message_json)
            message_id = future.result()
            logger.info(f"Published Pub/Sub message for task_id {task_id} with action '{action}': {message_id}")

        except Exception as e:
            logger.error(f"Failed to publish Pub/Sub message for task_id {task_id}: {e}")

    def execute(self) -> Dict[str, Any]:
        """Execute training using a SINGLE TrainPipeline over all datasets.

        The previous implementation created one pipeline per dataset and aggregated
        their results. The TrainPipeline already supports multi-dataset mode by
        accepting a list of dataset names. We now:
        - If dataset_address is provided: extract dataset name and use it (single dataset mode)
        - Otherwise: Discover all datasets (optionally truncated by n_datasets_to_use)
        - Instantiate exactly one TrainPipeline whose pipeline_name (and external
          identity) is self.pipeline_name
        - Pass either a single dataset string or list of dataset strings to .run()
        - Return a flattened summary without per-dataset training_results
        """
        logger.info(f"--- Starting Training Workflow (single-pipeline) Tenant: {self.tenant_id} ---")
        logger.info(f"Training configuration: {self.training_params.dict() if self.training_params else {}}")
        logger.info(f"Model configuration: {self.model_params.dict() if self.model_params else {}}")

        # Check for cancellation before starting
        if self.cancellation_event and self.cancellation_event.is_set():
            logger.info("Training cancelled before execution")
            result = {
                "status": "cancelled",
                "datasets_found": 0,
                "steps_completed": 0,
                "run_id": self.pipeline_name,
                "pipeline_name": self.pipeline_name,
                "run_guids": [],
                "message": "Training cancelled before execution",
                "dataset_mode": "none",
                # Compatibility fields
                "total_runs": 0,
                "successful_runs": 0,
                "training_results": [],
            }
            self._update_firestore_status("cancelled")
            return result

        try:
            # Determine dataset(s) to use
            if self.dataset_address:
                # Use provided dataset address
                logger.info(f"Using provided dataset address: {self.dataset_address}")
                try:
                    dataset_name = self._extract_dataset_name_from_path(self.dataset_address)
                    datasets = [dataset_name]
                    logger.info(f"Extracted dataset name: {dataset_name}")
                except ValueError as e:
                    logger.error(f"Invalid dataset address: {e}")
                    result = {
                        "status": "failed",
                        "datasets_found": 0,
                        "steps_completed": 0,
                        "run_id": self.pipeline_name,
                        "pipeline_name": self.pipeline_name,
                        "run_guids": [],
                        "message": f"Invalid dataset address: {e}",
                        "dataset_mode": "none",
                        "error": str(e),
                        # Compatibility fields
                        "total_runs": 0,
                        "successful_runs": 0,
                        "training_results": [],
                    }
                    self._update_firestore_status("error", str(e))
                    return result
            else:
                # Discover datasets
                datasets = self.discover_datasets()
            
            if not datasets:
                logger.warning("No datasets found for training")
                result = {
                    "status": "completed",
                    "datasets_found": 0,
                    "steps_completed": 0,
                    "run_id": self.pipeline_name,
                    "pipeline_name": self.pipeline_name,
                    "run_guids": [],
                    "message": "No datasets available for training",
                    "dataset_mode": "none",
                    # Compatibility fields
                    "total_runs": 0,
                    "successful_runs": 0,
                    "training_results": [],
                }
                self._update_firestore_status("completed")
                return result

            dataset_mode = "multi" if len(datasets) > 1 else "single"

            # Merge params for TrainPipeline (avoid duplicate pipeline_name)
            # Convert Pydantic models to dicts for TrainPipeline
            all_kwargs = {
                **(self.training_params.dict() if self.training_params else {}),
                **(self.model_params.dict() if self.model_params else {}),
                **(self.eval_params.dict() if self.eval_params else {})
            }
            all_kwargs.pop("pipeline_name", None)

            # Create single pipeline; override run_guid with external identity (task_id/pipeline_name)
            pipeline_identity = self.task_id or self.pipeline_name or "training_run"
            train_pipeline = TrainPipeline(
                tenant_id=self.tenant_id,
                verbose=self.verbose,
                pipeline_name=pipeline_identity,  # Also used as identity
                run_guid=pipeline_identity,
                **all_kwargs
            )

            # Execute once over all datasets (list or single item)
            dataset_arg = datasets if dataset_mode == "multi" else datasets[0]

            # Check for cancellation before pipeline execution
            if self.cancellation_event and self.cancellation_event.is_set():
                logger.info("Training cancelled before pipeline execution")
                result = {
                    "status": "cancelled",
                    "datasets_found": len(datasets),
                    "steps_completed": 0,
                    "run_id": pipeline_identity,
                    "pipeline_name": pipeline_identity,
                    "run_guids": [],
                    "message": "Training cancelled before pipeline execution",
                    "dataset_mode": dataset_mode
                }
                self._update_firestore_status("cancelled")
                return result

            # Add task_id to wandb tags if provided
            wandb_tags = self.wandb_tags.copy()
            if self.task_id:
                wandb_tags.append(f"task_id:{self.task_id}")

            # Update status to running before starting training
            if self.task_id:
                self._update_firestore_status("running")

            pipeline_result = train_pipeline.run(
                dataset_name=dataset_arg,
                wandb_run_tags=wandb_tags,
                custom_name=self.custom_name,
            )

            status = pipeline_result.get("status", "unknown")
            steps_completed = int(pipeline_result.get("steps_completed", 0))
            steps_failed = int(pipeline_result.get("steps_failed", 0))

            # Back-compat summary across (now single) run
            total_runs = 1
            successful_runs = 1 if (status.lower() == "completed" and steps_failed == 0) else 0

            final_status = "completed" if status == "completed" else status
            result = {
                "status": final_status,
                "datasets_found": len(datasets),
                "steps_completed": steps_completed,
                "run_id": pipeline_identity,
                "pipeline_name": pipeline_identity,
                "run_guids": [train_pipeline.run_guid],  # type: ignore[attr-defined]
                "dataset_mode": dataset_mode,
                "pipeline_result": pipeline_result,
                # Compatibility fields expected by CLI
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "training_results": [],
            }
            self._update_firestore_status(final_status)
            return result

        except InterruptedError:
            logger.info("Training workflow cancelled (single-pipeline mode)")
            
            # Check if cancellation was due to timeout
            is_timeout = self.timeout_triggered and self.timeout_triggered.is_set()
            
            if is_timeout:
                logger.info("Cancellation was triggered by execution timeout - initiating auto-resume")
                # Publish auto-resume message to Pub/Sub
                self._publish_auto_resume_message()
                # Update Firestore status to auto_suspended
                self._update_firestore_status("auto_suspended", f"Auto-suspended at attempt #{self.auto_resume_count + 1}")
                result = {
                    "status": "auto_suspended",
                    "datasets_found": 0,
                    "steps_completed": 0,
                    "run_id": self.pipeline_name,
                    "pipeline_name": self.pipeline_name,
                    "run_guids": [self.pipeline_name],
                    "dataset_mode": "unknown",
                    "message": f"Training auto-suspended for resume (attempt #{self.auto_resume_count + 1})",
                    # Compatibility fields
                    "total_runs": 0,
                    "successful_runs": 0,
                    "training_results": [],
                }
                return result
            else:
                # Normal user cancellation
                result = {
                    "status": "cancelled",
                    "datasets_found": 0,
                    "steps_completed": 0,
                    "run_id": self.pipeline_name,
                    "pipeline_name": self.pipeline_name,
                    "run_guids": [self.pipeline_name],
                    "custom_name": self.custom_name,
                    "dataset_mode": "unknown",
                    "error": "Training workflow cancelled by user request",
                    # Compatibility fields
                    "total_runs": 0,
                    "successful_runs": 0,
                    "training_results": [],
                }
                self._update_firestore_status("cancelled", "Training workflow cancelled by user request")
                return result
        except Exception as e:
            logger.error(f"Training workflow failed: {e}")
            result = {
                "status": "failed",
                "datasets_found": 0,
                "steps_completed": 0,
                "run_id": self.pipeline_name,
                "pipeline_name": self.pipeline_name,
                "run_guids": [self.pipeline_name],
                "custom_name": self.custom_name,
                "dataset_mode": "unknown",
                "error": str(e),
                # Compatibility fields
                "total_runs": 0,
                "successful_runs": 0,
                "training_results": [],
            }
            self._update_firestore_status("error", str(e))
            return result

    # Removed _run_training_for_dataset: single pipeline now handles all datasets.


def train_workflow(tenant_id: str, **kwargs):
    """
    Convenience function for running training workflow.

    This function provides a simple interface for running the training workflow
    with keyword arguments, similar to the original train_all.train() function.

    Args:
        tenant_id: The tenant ID for GCS operations.
        **kwargs: Additional keyword arguments for TrainingWorkflow.

    Returns:
        Dictionary containing training results.
    """
    workflow = TrainingWorkflow(tenant_id=tenant_id, **kwargs)
    return workflow.execute()
