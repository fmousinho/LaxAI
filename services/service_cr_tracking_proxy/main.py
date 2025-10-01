"""
Pub/Sub to Cloud Run Job Proxy for the LaxAI Tracking Service.

This Cloud Function is triggered by messages on a Pub/Sub topic. It executes the existing
'laxai-service-tracking' Google Cloud Run Job (v2 API) with customized arguments based on
the message content for tracking tasks.

Environment Variables:
    - GOOGLE_CLOUD_PROJECT: Automatically populated by GCP with the project ID.
    - CLOUD_REGION: Must be set manually (defaults to 'us-central1' if not provided).
"""

import base64
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import pubsub_v1
from google.cloud import firestore  # type: ignore
from google.cloud.run_v2 import JobsClient, ExecutionsClient
from google.cloud.run_v2.types import Job, RunJobRequest, DeleteJobRequest, CancelExecutionRequest, Execution

# Centralized logging configuration
from shared_libs.config import logging_config  # noqa: F401

logger = logging.getLogger(__name__)

_JOB_NAME = "laxai-service-tracking"
_CLOUD_REGION_DEFAULT = "us-central1"

class TrackingJobProxy:
    def get_tracking_job_status(self, task_id: str) -> dict:
        """
        Retrieve the status and metadata for a tracking job by task_id from Firestore.
        Returns a dict with status, error, operation_name, execution_name, timestamps, etc.
        """
        try:
            doc = self._progress_collection.document(task_id).get()
            if not doc.exists:
                return {"task_id": task_id, "status": "not_found", "error": "No mapping found for this task_id."}
            data = doc.to_dict() or {}
            # Optionally filter/format fields for API response
            return data
        except Exception as e:
            return {"task_id": task_id, "status": "error", "error": str(e)}

    """Handles the creation and execution of Cloud Run Jobs for tracking."""

    def __init__(self):
        """Initializes the proxy with GCP clients and configuration."""
        # These are automatically populated by GCP
        self.project_id = os.environ["GOOGLE_CLOUD_PROJECT"]

        # CLOUD_REGION needs to be set manually in function configuration
        self.region = os.environ.get("CLOUD_REGION", _CLOUD_REGION_DEFAULT)

        # Cloud Run Jobs client and resource parent path
        self.run_client = JobsClient()
        self.executions_client = ExecutionsClient()
        self.parent = f"projects/{self.project_id}/locations/{self.region}"

        # Firestore client for mapping task_id -> operation/execution
        self.db = firestore.Client(project=self.project_id)
        self._progress_collection = self.db.collection("tracking_progress")

    def _build_run_request(self, payload: Dict[str, Any]) -> RunJobRequest:
        """Builds a RunJobRequest for the existing laxai-service-tracking job with args overrides.
        
        This method leverages the pre-configured laxai-service-tracking job and only overrides
        the container arguments to customize the tracking run.
        """
        tenant_id = payload.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required in the payload")
        custom_name = payload.get("custom_name", "tracking-run")

        # Build command arguments
        task_id = payload.get("task_id")
        args = [
            f"--tenant_id={tenant_id}",
            f"--custom_name={custom_name}",
            *( [f"--task_id={task_id}"] if task_id else [] ),
        ]

        # Add tracking-specific parameters
        for key, value in payload.get("tracking_params", {}).items():
            if value is not None:
                # Convert underscores to dashes for CLI argument format
                cli_arg_name = key.replace('_', '-')
                
                # Handle boolean parameters as flags
                if isinstance(value, bool):
                    if value:
                        args.append(f"--{cli_arg_name}")
                        logger.info(f"Added boolean flag: --{cli_arg_name} (from tracking_params.{key})")
                    else:
                        logger.info(f"Skipped boolean flag: --{cli_arg_name} (False value from tracking_params.{key})")
                else:
                    args.append(f"--{cli_arg_name}={value}")
                    logger.info(f"Added argument: --{cli_arg_name}={value} (from tracking_params.{key})")

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

    def start_tracking_job(self, payload: Dict[str, Any]) -> Optional[str]:
        """Creates and executes a tracking job asynchronously using the existing laxai-service-tracking job."""

        try:
            # Build run request for existing job with args override
            run_request = self._build_run_request(payload)

            # Execute the existing job asynchronously with overrides
            operation = self.run_client.run_job(request=run_request)
            if operation.metadata is not None:
                execution_name = operation.metadata.name
                logger.info(f"Started job - Execution Name: {execution_name}")
            else:
                execution_name = None
                logger.error("Operation metadata is None; cannot extract execution name.")
        
            task_id = payload.get("task_id")
            if task_id:
                now = datetime.now(timezone.utc).isoformat()
                doc = {
                    "task_id": task_id,
                    "execution_name": execution_name,
                    "job_name": _JOB_NAME,
                    "region": self.region,
                    "status": "not_started",  # Use PipelineStatus.NOT_STARTED value
                    "created_at": now,
                    "updated_at": now,
                    "error": None,
                    # Flatten payload fields (exclude action and task_id to avoid duplication)
                    "tenant_id": payload.get("tenant_id"),
                    "custom_name": payload.get("custom_name"),
                    "tracking_params": payload.get("tracking_params")
                }
                try:
                    doc_ref = self._progress_collection.document(task_id)
                    if not doc_ref.get().exists:
                        doc_ref.set(doc)
                        logger.info(f"Persisted run mapping for task_id={task_id} -> operation={execution_name}")
                    else:
                        logger.warning(f"Mapping for task_id={task_id} already exists; not overwriting")
                except Exception as fe:
                    logger.error(f"Failed to persist run mapping for task_id={task_id}: {fe}")

        except GoogleAPIError as e:
            logger.error(f"Failed to start tracking job: {e}")
            raise
        
        return task_id
              

    def cancel_tracking_job(self, task_id: str) -> None:
        """Cancel a running tracking task using persisted mapping.

        Uses the execution_name stored in Firestore for direct cancellation.
        """
        try:
            doc_ref = self._progress_collection.document(task_id)
            doc = doc_ref.get()
            if not doc.exists:
                logger.warning(f"No mapping found for task_id={task_id}; nothing to cancel")
                return
            data = doc.to_dict() or {}
            execution_name = data.get("execution_name")

            logger.info(f"Cancellation attempt for task_id={task_id}: execution_name={execution_name}")

            if not execution_name:
                logger.warning(f"No execution_name stored for task_id={task_id}; cannot cancel directly")
                logger.info(f"Stored data: {data}")
                return

            # Cancel the execution directly using the stored execution_name
            try:
                operation = self.executions_client.cancel_execution(name=execution_name, timeout=60)
                results = operation.result()  # Wait for operation to complete
                
                logger.info(f"Successfully requested cancellation for execution {execution_name} (task_id={task_id})")
                
                # Update status to indicate cancellation was requested
                doc_ref.update({"status": "cancelled", "updated_at": datetime.now(timezone.utc).isoformat() + "Z"})
                
            except GoogleAPIError as cancel_error:
                logger.error(f"Failed to cancel execution {execution_name}: {cancel_error}")
                doc_ref.update({"status": "error", "error": str(cancel_error), "updated_at": datetime.now(timezone.utc).isoformat() + "Z"})
                return  # Don't re-raise, just return to indicate cancellation attempt was made
            except Exception as cancel_error:
                logger.error(f"Unexpected error during cancellation of execution {execution_name}: {cancel_error}")
                doc_ref.update({"status": "error", "error": str(cancel_error), "updated_at": datetime.now(timezone.utc).isoformat() + "Z"})
                return  # Don't re-raise, just return to indicate cancellation attempt was made
        
        except Exception as e:
            logger.error(f"Failed to cancel task_id={task_id}: {e}")
            # Try to update mapping with error
            try:
                self._progress_collection.document(task_id).update({"status": "error", "error": str(e), "updated_at": datetime.now(timezone.utc).isoformat() + "Z"})
            except Exception:
                pass
            # Don't re-raise the exception, just log it

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
                job_id = self.start_tracking_job(payload)

                logger.info(f"Queued tracking job: {job_id}")

            elif action == "cancel":
                task_id = payload.get("task_id")
                if not task_id:
                    raise ValueError("Cancel action requires task_id")
                self.cancel_tracking_job(task_id)
                logger.info(f"Cancel request queued for task_id: {task_id}")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid message format: {e}")
            raise
        except GoogleAPIError as e:
            logger.error(f"API error processing message: {e}")
            raise


def process_pubsub_message(event, context):
    """Cloud Function entry point for Pub/Sub-triggered tracking jobs.

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

        proxy = TrackingJobProxy()
        proxy.process_message(message_data)

        logger.info("Successfully processed Pub/Sub message")

    except Exception as e:
        logger.error(f"Error in process_pubsub_message: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise