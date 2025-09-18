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
from typing import Any, Dict

from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import pubsub_v1
from google.cloud.run_v2 import JobsClient
from google.cloud.run_v2.types import Job, RunJobRequest, DeleteJobRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_JOB_NAME = "laxai-service-training"
_CLOUD_REGION_DEFAULT = "us-central1"

class TrainingJobProxy:
    """Handles the creation and execution of Cloud Run Jobs for training."""

    def __init__(self):
        """Initializes the proxy with GCP clients and configuration."""
        # These are automatically populated by GCP
        self.project_id = os.environ["GOOGLE_CLOUD_PROJECT"]

        # CLOUD_REGION needs to be set manually in function configuration
        self.region = os.environ.get("CLOUD_REGION", _CLOUD_REGION_DEFAULT)

        self.run_client = JobsClient()
        self.parent = f"projects/{self.project_id}/locations/{self.region}"

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
        args = [f"--tenant_id={tenant_id}", f"--custom_name={custom_name}"]

        if payload.get("resume_from_checkpoint", True):
            args.append("--resume_from_checkpoint")

        # Add dynamic parameters
        for param_group in ["training_params", "model_params", "eval_params"]:
            for key, value in payload.get(param_group, {}).items():
                if value is not None:
                    args.append(f"--{key}={value}")

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

        try:
            # Build run request for existing job with args override
            run_request = self._build_run_request(payload)

            # Execute the existing job asynchronously with overrides
            operation = self.run_client.run_job(request=run_request)
            logger.info(f"Started execution of existing job 'laxai-service-training' with ID: {job_id}, operation: {operation.operation.name}")

            # Don't wait for completion - fire and forget
            return job_id

        except GoogleAPIError as e:
            logger.error(f"Failed to start training job {job_id}: {e}")
            raise

    def cancel_training_job(self, job_name: str) -> None:
        """Cancels a running training job by deleting it."""
        full_job_name = f"{self.parent}/jobs/{job_name}"
        try:
            delete_request = DeleteJobRequest(name=full_job_name)
            operation = self.run_client.delete_job(request=delete_request)
            operation.result()  # Wait for deletion
            logger.info(f"Cancelled and deleted training job: {job_name}")

        except NotFound:
            logger.warning(f"Job {job_name} not found - may already be cancelled")
        except GoogleAPIError as e:
            logger.error(f"Failed to cancel job {job_name}: {e}")
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
                job_name = payload.get("job_name")
                if not job_name:
                    raise ValueError("Cancel action requires job_name")
                self.cancel_training_job(job_name)
                logger.info(f"Cancelled job: {job_name}")

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