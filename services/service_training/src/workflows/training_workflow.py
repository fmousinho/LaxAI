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

from training_controller import TrainingController

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
        """Execute training using TrainingController."""
        logger.info(f"--- Starting Training Workflow Tenant: {self.tenant_id} ---")
        logger.info(f"Training configuration: {self.training_params.dict() if self.training_params else {}}")
        logger.info(f"Evaluation configuration: {self.eval_params.dict() if self.eval_params else {}}")
        
        # Check for cancellation before starting
        if self.cancellation_event and self.cancellation_event.is_set():
            logger.info("Training cancelled before execution")
            result = {
                "status": "cancelled",
                "message": "Training cancelled before execution",
            }
            self._update_firestore_status("cancelled")
            return result

        try:
            # Determine dataset(s) to use
            datasets = []
            
            # Check if dataset_address is provided directly OR in training_params
            provided_address = self.dataset_address
            if not provided_address and self.training_params and self.training_params.dataset_address:
                provided_address = self.training_params.dataset_address

            if provided_address:
                # Use provided dataset address
                logger.info(f"Using provided dataset address: {provided_address}")
                if isinstance(provided_address, str):
                    datasets = [provided_address]
                else:
                    datasets = provided_address
            else:
                # Discover datasets
                if hasattr(self, 'discover_datasets'):
                    datasets = self.discover_datasets()
                else:
                    logger.warning("discover_datasets method not found on TrainingWorkflow instance. No datasets will be used unless dataset_address is provided.")
            
            if not datasets:
                logger.warning("No datasets found for training")
                result = {
                    "status": "completed",
                    "message": "No datasets available for training",
                }
                self._update_firestore_status("completed")
                return result

            # Update training_params with discovered datasets
            if self.training_params:
                # Assuming TrainingParams has a dataset_address field that can take a list of strings
                self.training_params.dataset_address = datasets

            # Initialize TrainingController
            controller = TrainingController(
                tenant_id=self.tenant_id,
                wandb_run_name=self.wandb_run_name,
                training_params=self.training_params,
                eval_params=self.eval_params,
                task_id=self.task_id, # Pass task_id to controller for internal logging/tracking
            )

            # Check for cancellation before training
            if self.cancellation_event and self.cancellation_event.is_set():
                logger.info("Training cancelled before controller execution")
                self._update_firestore_status("cancelled")
                return {"status": "cancelled", "message": "Training cancelled before controller execution"}

            # We can start a thread to monitor cancellation_event and update controller
            def monitor_cancellation():
                if self.cancellation_event:
                    self.cancellation_event.wait()
                    if self.cancellation_event.is_set():
                        logger.info("Cancellation event detected, requesting graceful cancellation from TrainingController.")
                        controller.graceful_cancellation_request(self.task_id or "unknown")
            
            monitor_thread = threading.Thread(target=monitor_cancellation, daemon=True)
            monitor_thread.start()

            # Update status to running before starting training
            if self.task_id:
                self._update_firestore_status("running")

            task_id = controller.train()

            # Check if training was cancelled (graceful shutdown)
            if self.cancellation_event and self.cancellation_event.is_set():
                logger.info("Training finished due to cancellation event")
                
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
                        "task_id": task_id,
                        "run_id": self.wandb_run_name,
                        "message": f"Training auto-suspended for resume (attempt #{self.auto_resume_count + 1})",
                    }
                    return result
                else:
                    # Normal user cancellation
                    logger.info("Cancellation was triggered by user request")
                    self._update_firestore_status("cancelled", "Training workflow cancelled by user request")
                    result = {
                        "status": "cancelled",
                        "task_id": task_id,
                        "run_id": self.wandb_run_name,
                        "message": "Training workflow cancelled by user request",
                    }
                    return result

            result = {
                "status": "completed",
                "task_id": task_id,
                "run_id": self.wandb_run_name, 
                "message": f"Training completed successfully for task_id: {task_id}"
            }
            self._update_firestore_status("completed")
            return result

        except InterruptedError:
            logger.info("Training workflow cancelled")
            
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
                    "message": f"Training auto-suspended for resume (attempt #{self.auto_resume_count + 1})",
                }
                return result
            else:
                # Normal user cancellation
                result = {
                    "status": "cancelled",
                    "error": "Training workflow cancelled by user request",
                }
                self._update_firestore_status("cancelled", "Training workflow cancelled by user request")
                return result
        except Exception as e:
            logger.error(f"Training workflow failed: {e}")
            result = {
                "status": "failed",
                "error": str(e),
            }
            self._update_firestore_status("error", str(e))
            return result



