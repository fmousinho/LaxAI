"""
Pub/Sub to Cloud Run Job Proxy for the LaxAI Training Service.

This Cloud Function is triggered by messages on a Pub/Sub topic. It executes the existing
'laxai-service-training' Google Cloud Run Job (v2 API) with customized arguments based on
the message content for training tasks.

Environment Variables:
    - GOOGLE_CLOUD_PROJECT: Automatically populated by GCP with the project ID.
    - CLOUD_REGION: Must be set manually (defaults to 'us-central1' if not provided).
"""

import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import pubsub_v1
from google.cloud import firestore  # type: ignore
from google.cloud.run_v2 import JobsClient
from google.cloud.run_v2.types import Job, RunJobRequest, DeleteJobRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_JOB_NAME = "laxai-service-training"
_CLOUD_REGION_DEFAULT = "us-central1"

class TrainingJobProxy:
    def get_training_job_status(self, task_id: str) -> dict:
        """
        Retrieve the status and metadata for a training job by task_id from Firestore.
        Returns a dict with status, error, operation_name, execution_name, timestamps, etc.
        """
        try:
            doc = self._runs_collection.document(task_id).get()
            if not doc.exists:
                return {"task_id": task_id, "status": "not_found", "error": "No mapping found for this task_id."}
            data = doc.to_dict() or {}
            # Optionally filter/format fields for API response
            return data
        except Exception as e:
            return {"task_id": task_id, "status": "error", "error": str(e)}
    """Handles the creation and execution of Cloud Run Jobs for training."""

    def __init__(self):
        """Initializes the proxy with GCP clients and configuration."""
        # These are automatically populated by GCP
        self.project_id = os.environ["GOOGLE_CLOUD_PROJECT"]

        # CLOUD_REGION needs to be set manually in function configuration
        self.region = os.environ.get("CLOUD_REGION", _CLOUD_REGION_DEFAULT)

        # Cloud Run Jobs client and resource parent path
        self.run_client = JobsClient()
        self.parent = f"projects/{self.project_id}/locations/{self.region}"

        # Firestore client for mapping task_id -> operation/execution
        self.db = firestore.Client(project=self.project_id)
        self._runs_collection = self.db.collection("training_runs")

    def _build_run_request(self, payload: Dict[str, Any]) -> RunJobRequest:
        """Builds a RunJobRequest for the existing laxai-service-training job with args overrides.
        
        This method leverages the pre-configured laxai-service-training job and only overrides
        the container arguments to customize the training run.
        """
        tenant_id = payload.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required in the payload")
        custom_name = payload.get("custom_name", "training-run")

        # Build command arguments
        task_id = payload.get("task_id")
        args = [
            f"--tenant_id={tenant_id}",
            f"--custom_name={custom_name}",
            *( [f"--task_id={task_id}"] if task_id else [] ),
        ]

        if payload.get("resume_from_checkpoint", True):
            args.append("--resume_from_checkpoint")

        # Add dynamic parameters
        for param_group in ["training_params", "model_params", "eval_params"]:
            for key, value in payload.get(param_group, {}).items():
                if value is not None:
                    # Handle parameter name mapping for different param groups
                    if param_group == "eval_params" and key == "prefetch_factor":
                        arg_name = "eval_prefetch_factor"
                    elif param_group == "training_params" and key == "prefetch_factor":
                        arg_name = "train_prefetch_factor"
                    else:
                        arg_name = key
                    args.append(f"--{arg_name}={value}")
                    logger.info(f"Added argument: --{arg_name}={value} (from {param_group}.{key})")

        # Use the existing job with args override
        job_name = f"{self.parent}/jobs/{_JOB_NAME}"

        run_request = RunJobRequest(
            name=job_name,
            overrides={
                "container_overrides": [{
                    "args": args
                }]
            }
        )

        return run_request

    def start_training_job(self, payload: Dict[str, Any]) -> str:
        """Creates and executes a training job asynchronously using the existing laxai-service-training job."""
        custom_name = payload.get("custom_name", "training-run")
        job_id = f"{custom_name}-{uuid.uuid4().hex[:8]}"

        import datetime
        try:
            # Build run request for existing job with args override
            run_request = self._build_run_request(payload)

            # Execute the existing job asynchronously with overrides
            operation = self.run_client.run_job(request=run_request)
            op_name: Optional[str] = getattr(getattr(operation, "operation", None), "name", None)
            # Try to extract execution_name from operation metadata if available (optional)
            execution_name = None
            if hasattr(operation, "metadata") and hasattr(operation.metadata, "execution"):
                execution_name = getattr(operation.metadata, "execution", None)
            logger.info(f"Started execution of existing job 'laxai-service-training' with ID: {job_id}, operation: {op_name}, execution: {execution_name}")

            # Persist mapping task_id -> operation name (and metadata snapshot)
            task_id = payload.get("task_id")
            if task_id:
                now = datetime.datetime.utcnow().isoformat() + "Z"
                doc = {
                    "task_id": task_id,
                    "operation_name": op_name,
                    "execution_name": execution_name,
                    "job_name": _JOB_NAME,
                    "region": self.region,
                    "status": "running",
                    "created_at": now,
                    "updated_at": now,
                    "error": None,
                    "payload": {k: v for k, v in payload.items() if k != "eval_params"},
                }
                try:
                    # Idempotent: only create if not exists
                    doc_ref = self._runs_collection.document(task_id)
                    if not doc_ref.get().exists:
                        doc_ref.set(doc)
                        logger.info(f"Persisted run mapping for task_id={task_id} -> operation={op_name}")
                    else:
                        logger.warning(f"Mapping for task_id={task_id} already exists; not overwriting")
                except Exception as fe:
                    logger.error(f"Failed to persist run mapping for task_id={task_id}: {fe}")

            # Don't wait for completion - fire and forget
            return job_id

        except GoogleAPIError as e:
            logger.error(f"Failed to start training job {job_id}: {e}")
            # Update mapping with error if possible
            task_id = payload.get("task_id")
            if task_id:
                try:
                    self._runs_collection.document(task_id).update({"status": "error", "error": str(e), "updated_at": datetime.datetime.utcnow().isoformat() + "Z"})
                except Exception:
                    pass
            raise

    def cancel_training_job(self, task_id: str) -> None:
        """Cancel a running training task using persisted mapping (best-effort LRO cancel).

        This cancels the Long-Running Operation if available and updates the mapping status.
        """
        import datetime
        try:
            doc_ref = self._runs_collection.document(task_id)
            doc = doc_ref.get()
            if not doc.exists:
                logger.warning(f"No mapping found for task_id={task_id}; nothing to cancel")
                return
            data = doc.to_dict() or {}
            op_name = data.get("operation_name")
            if not op_name:
                logger.warning(f"No operation_name stored for task_id={task_id}; cannot cancel LRO")
                return

            # Access the operations client from the transport to cancel the operation
            operations_client = self.run_client.transport.operations_client
            try:
                operations_client.cancel_operation(name=op_name)
                logger.info(f"Requested cancellation for operation {op_name} (task_id={task_id})")
                doc_ref.update({"status": "cancelling", "updated_at": datetime.datetime.utcnow().isoformat() + "Z"})
            except Exception as ce:
                logger.error(f"Error cancelling operation {op_name}: {ce}")
                doc_ref.update({"status": "error", "error": str(ce), "updated_at": datetime.datetime.utcnow().isoformat() + "Z"})
        except Exception as e:
            logger.error(f"Failed to cancel task_id={task_id}: {e}")
            # Try to update mapping with error
            try:
                self._runs_collection.document(task_id).update({"status": "error", "error": str(e), "updated_at": datetime.datetime.utcnow().isoformat() + "Z"})
            except Exception:
                pass
            raise

    def process_message(self, message_data: str) -> None:
        """Processes a decoded Pub/Sub message."""
        try:
            payload = json.loads(message_data)
            logger.info(f"Parsed payload: {payload}")

            action = payload.get("action")
            if not action:
                raise ValueError("Action is required in the payload")

            if action not in ["create", "cancel"]:
                raise ValueError(f"Invalid action: {action}")

            if action == "create":
                tenant_id = payload.get("tenant_id")
                if not tenant_id:
                    logger.error(f"Missing tenant_id in payload. Available keys: {list(payload.keys())}")
                    raise ValueError("Create action requires tenant_id")
                job_id = self.start_training_job(payload)
                logger.info(f"Queued training job: {job_id}")

            elif action == "cancel":
                task_id = payload.get("task_id")
                if not task_id:
                    raise ValueError("Cancel action requires task_id")
                self.cancel_training_job(task_id)
                logger.info(f"Cancel request queued for task_id: {task_id}")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid message format: {e}")
            raise
        except GoogleAPIError as e:
            logger.error(f"API error processing message: {e}")
            raise


def process_pubsub_message(event, context):
    """Cloud Function entry point for Pub/Sub-triggered training jobs.

    Args:
        event: Pub/Sub event data containing:
            - data: Base64-encoded message payload
            - attributes: Optional message attributes dict
            - messageId: Unique message identifier
            - publishTime: ISO 8601 timestamp
            - orderingKey: For ordered messages (optional)
        context: Event context containing:
            - eventId: Unique event identifier
            - timestamp: Event timestamp
            - eventType: Event type (google.pubsub.topic.publish)
            - resource: Resource information
    """
    try:
        logger.info(f"Received event: {event}")
        logger.info(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")

        if 'data' not in event:
            logger.error("Missing 'data' in Pub/Sub event")
            return

        message_data = base64.b64decode(event['data']).decode('utf-8')
        logger.info(f"Decoded message data: {message_data}")

        proxy = TrainingJobProxy()
        proxy.process_message(message_data)

        logger.info("Successfully processed Pub/Sub message")

    except Exception as e:
        logger.error(f"Error in process_pubsub_message: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise