"""
Pub/Sub to Cloud Run Job Proxy for the LaxAI Training Service.

This Cloud Function is triggered by messages on a Pub/Sub topic. It creates and
executes Google Cloud Run Jobs based on the message content for training tasks.

Required Environment Variables:
    - GOOGLE_CLOUD_PROJECT: The GCP project ID.
    - CLOUD_REGION: The region for Cloud Run Jobs (e.g., 'us-central1').
    - JOB_DOCKER_IMAGE_URI: The full URI of the Docker image for the training job (optional, has default).
"""

import base64
import json
import logging
import os
import uuid
from typing import Any, Dict

from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import pubsub_v1
from google.cloud.run_v1 import JobsClient
from google.cloud.run_v1.types import Job, TaskTemplate, Container, CreateJobRequest, DeleteJobRequest, ExecuteJobRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingJobProxy:
    """Handles the creation and execution of Cloud Run Jobs for training."""

    def __init__(self):
        """Initializes the proxy with GCP clients and configuration."""
        self.project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
        self.region = os.environ["CLOUD_REGION"]
        self.image_uri = os.getenv(
            "JOB_DOCKER_IMAGE_URI",
            f"{self.region}-docker.pkg.dev/{self.project_id}/laxai-repo/laxai-service-training:latest"
        )

        self.run_client = JobsClient()
        self.parent = f"projects/{self.project_id}/locations/{self.region}"

        # Default training command
        self.training_command = [
            "python", "services/service_training/src/cli/train_cli.py"
        ]

    def _build_job_spec(self, payload: Dict[str, Any]) -> Job:
        """Builds a Cloud Run Job specification from the payload."""
        tenant_id = payload.get("tenant_id", "default")
        custom_name = payload.get("custom_name", "training-run")

        # Resource configuration with defaults
        resources_config = payload.get("resources", {})
        cpu = resources_config.get("cpu", "4")
        memory = resources_config.get("memory", "16Gi")
        gpu_count = resources_config.get("gpu_count", 1)  # Default to 0 GPUs
        timeout_seconds = payload.get("timeout_seconds", 36000)  # 10 hours

        # Build command arguments
        args = [f"--tenant_id={tenant_id}", f"--custom_name={custom_name}"]

        if payload.get("resume_from_checkpoint", True):
            args.append("--resume_from_checkpoint")

        # Add dynamic parameters
        for param_group in ["training_params", "model_params", "eval_params"]:
            for key, value in payload.get(param_group, {}).items():
                if value is not None:
                    args.append(f"--{key}={value}")

        # Container resources
        resources = {
            "limits": {"cpu": cpu, "memory": memory},
            "startup_cpu_boost": True,
        }
        if gpu_count > 0:
            resources["limits"]["nvidia.com/gpu"] = str(gpu_count)

        container = Container(
            image=self.image_uri,
            command=self.training_command,
            args=args,
            resources=resources,
        )

        # Job specification
        job_spec = Job(
            template=TaskTemplate(
                containers=[container],
                timeout=f"{timeout_seconds}s",
                max_retries=2,
            )
        )
        return job_spec

    def start_training_job(self, payload: Dict[str, Any]) -> str:
        """Creates and executes a training job asynchronously."""
        custom_name = payload.get("custom_name", "training-run")
        job_id = f"{custom_name}-{uuid.uuid4().hex[:8]}"

        try:
            job_spec = self._build_job_spec(payload)

            # Create the job
            create_request = CreateJobRequest(
                parent=self.parent,
                job=job_spec,
                job_id=job_id
            )
            created_job = self.run_client.create_job(request=create_request)
            logger.info(f"Created training job: {created_job.name}")

            # Execute the job asynchronously
            execute_request = ExecuteJobRequest(name=created_job.name)
            operation = self.run_client.execute_job(request=execute_request)
            logger.info(f"Started execution of job: {job_id}, operation: {operation.operation.name}")

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
            action = payload.get("action")

            if action not in ["create", "cancel"]:
                raise ValueError(f"Invalid action: {action}")

            if action == "create":
                if "tenant_id" not in payload:
                    raise ValueError("Create action requires tenant_id")
                job_id = self.start_training_job(payload)
                logger.info(f"Queued training job: {job_id}")

            elif action == "cancel":
                if "job_name" not in payload:
                    raise ValueError("Cancel action requires job_name")
                self.cancel_training_job(payload["job_name"])
                logger.info(f"Cancelled job: {payload['job_name']}")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid message format: {e}")
            raise
        except GoogleAPIError as e:
            logger.error(f"API error processing message: {e}")
            raise


def process_pubsub_message(event, context):
    """Cloud Function entry point for Pub/Sub-triggered training jobs."""
    try:
        if 'data' not in event:
            logger.error("Missing 'data' in Pub/Sub event")
            return

        message_data = base64.b64decode(event['data']).decode('utf-8')
        logger.info(f"Processing Pub/Sub message: {message_data}")

        proxy = TrainingJobProxy()
        proxy.process_message(message_data)

        logger.info("Successfully processed Pub/Sub message")

    except Exception as e:
        logger.error(f"Error in process_pubsub_message: {e}")
        raise