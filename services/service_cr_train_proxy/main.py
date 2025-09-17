"""
Pub/Sub to Cloud Run Job Proxy for the LaxAI Training Service.

This Cloud Function is triggered by messages on a Pub/Sub topic. It creates or
cancels Google Cloud Run Jobs based on the message content.

Required Environment Variables:
    - GOOGLE_CLOUD_PROJECT: The GCP project ID.
    - CLOUD_REGION: The region for Cloud Run Jobs (e.g., 'us-central1').
    - JOB_DOCKER_IMAGE_URI: The full URI of the Docker image for the training job.
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
from google.cloud.run_v2.types import Job, TaskTemplate, Container, RunJobRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingJobProxy:
    """Handles the logic for creating and canceling Cloud Run Jobs."""

    def __init__(self):
        """Initializes the proxy and its clients."""
        # Required configuration from environment
        self.project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
        self.region = os.environ["CLOUD_REGION"]
        # Default image URI if not provided
        self.image_uri = os.getenv(
            "JOB_DOCKER_IMAGE_URI", 
            f"{self.region}-docker.pkg.dev/{self.project_id}/laxai-repo/laxai-service-training:latest"
        )
        
        # Initialize Google Cloud clients
        self.run_client = JobsClient()
        self.parent = f"projects/{self.project_id}/locations/{self.region}"
        
        # Command to run inside the training job container
        self.training_command = [
            "python", "services/service_training/src/cli/train_cli.py"
        ]

    def _parse_message_payload(self, message_data: str) -> Dict[str, Any]:
        """Parses and validates the JSON payload from a Pub/Sub message."""
        try:
            payload = json.loads(message_data)
            action = payload.get("action")

            if not action or action not in ["create", "cancel"]:
                raise ValueError("Payload must contain a valid 'action' ('create' or 'cancel').")

            if action == "create" and "tenant_id" not in payload:
                raise ValueError("A 'create' action requires a 'tenant_id'.")

            if action == "cancel" and "job_name" not in payload:
                raise ValueError("A 'cancel' action requires a 'job_name'.")
            
            return payload
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e}")
            raise ValueError(f"Invalid JSON in message: {e}") from e


    def _build_job_spec(self, payload: Dict[str, Any]) -> Job:
        """Constructs a Cloud Run Job object from the message payload."""
        tenant_id = payload.get("tenant_id", "default")
        custom_name = payload.get("custom_name", "training-run")
        
        # Allow resource configuration from the message, with sensible defaults
        resources = payload.get("resources", {})
        cpu = resources.get("cpu", "4")
        memory = resources.get("memory", "16Gi")
        gpu_count = resources.get("gpu_count", 1) # Default to 1 T4 GPU
        
        # Timeout in seconds, default to 10 hours
        timeout_seconds = payload.get("timeout_seconds", 36000)

        # Build container arguments from payload parameters
        args = [f"--tenant_id={tenant_id}", f"--custom_name={custom_name}"]
        
        # Add boolean flags like --resume_from_checkpoint if they are true
        if payload.get("resume_from_checkpoint", True):
            args.append("--resume_from_checkpoint")
        
        # Dynamically add other parameters
        for param_group in ["training_params", "model_params", "eval_params"]:
            for key, value in payload.get(param_group, {}).items():
                if value is not None:
                    args.append(f"--{key}={value}")

        resources = {
            "limits": {"cpu": cpu, "memory": memory},
            "startup_cpu_boost": True,
        }

        # Add GPU if requested
        if gpu_count > 0:
            resources["limits"]["nvidia.com/gpu"] = str(gpu_count)

        container = Container(
            image=self.image_uri,
            command=self.training_command,
            args=args,
            resources=resources,
        )

        job_spec = Job(
            template=TaskTemplate(
                containers=[container],
                timeout=f"{timeout_seconds}s",
                max_retries=2,
                parallelism=1
            )
        )
        return job_spec

    def start_training_job(self, payload: Dict[str, Any]) -> str:
        """Starts a training job in Cloud Run."""
        custom_name = payload.get("custom_name", "training-run")
        job_id = f"{custom_name}-{uuid.uuid4().hex[:8]}"

        try:
            job_spec = self._build_job_spec(payload)
            
            request = RunJobRequest(name=f"{self.parent}/jobs/{job_id}", overrides={})
            # Note: For one-off runs, RunJob is often better than CreateJob + ExecuteJob
            # It creates, runs, and optionally deletes the job. Here we create it.
            # A better approach is to have a single "template" job and execute it.
            # For this example, we create a unique job per request.
            
            created_job = self.run_client.create_job(parent=self.parent, job=job_spec, job_id=job_id)
            logger.info(f"Successfully created training job: {created_job.name}")
            
            # Now run the job
            operation = self.run_client.run_job(request=request)
            logger.info(f"Running job... Waiting for it to complete. Operation: {operation.metadata.name}")
            
            # The run_job call is long-running. In a real scenario you might not wait.
            # For a fire-and-forget proxy, you can just log the operation name and return.
            # response = operation.result()
            # logger.info(f"Job execution finished for: {response.name}")

            return job_id
        
        except GoogleAPIError as e:
            logger.error(f"Failed to create or run training job {job_id}: {e}")
            raise

    def cancel_training_job(self, job_name: str) -> None:
        """Cancels and deletes a Cloud Run Job."""
        full_job_name = f"{self.parent}/jobs/{job_name}"
        try:
            request = {"name": full_job_name}
            operation = self.run_client.delete_job(request=request)
            operation.result() # Wait for deletion to complete
            logger.info(f"Successfully deleted training job: {job_name}")
            
        except NotFound:
            logger.warning(f"Training job {job_name} not found. It may have already been deleted.")
        except GoogleAPIError as e:
            logger.error(f"Failed to delete training job {job_name}: {e}")
            raise
    
    def process_message(self, data: str):
        """Processes the decoded Pub/Sub message data."""
        payload = self._parse_message_payload(data)
        action = payload["action"]

        logger.info(f"Processing message with action: '{action}'")
        
        if action == "create":
            job_id = self.start_training_job(payload)
            logger.info(f"Initiated training job with ID: {job_id}")
        elif action == "cancel":
            job_name = payload["job_name"]
            self.cancel_training_job(job_name)


# Only instantiate the proxy when needed, not at module level
def process_pubsub_message(event, context):
    """Cloud Function entry point for processing Pub/Sub messages."""
    try:
        if 'data' not in event:
            logger.error("No 'data' field in the Pub/Sub message.")
            return

        message_data = base64.b64decode(event['data']).decode('utf-8')
        logger.info(f"Received message payload: {message_data}")
        
        # Instantiate proxy only when needed
        proxy = TrainingJobProxy()
        proxy.process_message(message_data)
        
        logger.info("Successfully processed Pub/Sub message.")

    except (ValueError, GoogleAPIError) as e:
        logger.error(f"Error processing Pub/Sub message: {e}")
        # Re-raise the exception to signal failure to Cloud Functions,
        # which will cause the message to be nack'd and retried.
        raise