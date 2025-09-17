"""
Pub/Sub Job Proxy for LaxAI Training Service.

This proxy listens to Pub/Sub messages on the "training-jobs" topic and
starts/cancels Google Cloud Run Jobs for training operations.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from google.api_core.exceptions import GoogleAPIError
from google.cloud import pubsub_v1, run_v1  #type: ignore
from google.protobuf import json_format
from google.pubsub_v1.types import PubsubMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TOPIC = "training-jobs"
_SUBSCRIPTION = "training-jobs-sub"



class TrainingJobProxy:
    """Proxy for handling training job requests via Pub/Sub."""

    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "laxai-466119")
        self.region = os.getenv("CLOUD_REGION", "us-central1")
        self.topic_name = _TOPIC
        self.subscription_name = os.getenv("PUBSUB_SUBSCRIPTION", _SUBSCRIPTION)


        # Cloud Run client
        self.run_client = run_v1.JobsClient()

        # Pub/Sub subscriber
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id, self.subscription_name
        )

        # Job configuration
        self.job_name = "laxai-training-job"
        self.image_uri = f"{self.region}-docker.pkg.dev/{self.project_id}/laxai-repo/laxai-service-training:latest"

        # Command to run the training CLI
        self.training_command = [
            "python", "services/service_training/src/cli/train_cli.py"
        ]

    def parse_message(self, message_data: bytes) -> Dict[str, Any]:
        """Parse Pub/Sub message data."""
        try:
            data = json.loads(message_data.decode('utf-8'))

            # Validate required fields
            if "action" not in data:
                raise ValueError("Message must contain 'action' field")

            if data["action"] not in ["create", "cancel"]:
                raise ValueError(f"Unknown action: {data['action']}")

            if data["action"] == "create" and "tenant_id" not in data:
                raise ValueError("Create action requires 'tenant_id' field")

            if data["action"] == "cancel" and "job_id" not in data:
                raise ValueError("Cancel action requires 'job_id' field")

            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            raise ValueError(f"Invalid JSON message: {e}")

    def create_job_request(self, message_data: Dict[str, Any]) -> run_v1.Job:
        """Create a Cloud Run Job request from message data."""

        # Extract parameters
        tenant_id = message_data.get("tenant_id", "default")
        custom_name = message_data.get("custom_name", "training_run")
        training_params = message_data.get("training_params", {})
        model_params = message_data.get("model_params", {})
        eval_params = message_data.get("eval_params", {})
        resume_from_checkpoint = message_data.get("resume_from_checkpoint", True)

        # Build container arguments
        args = [
            "--tenant_id", tenant_id,
            "--custom_name", custom_name,
        ]

        if resume_from_checkpoint:
            args.append("--resume_from_checkpoint")

        # Add training parameters
        for key, value in training_params.items():
            if value is not None:
                args.extend([f"--{key}", str(value)])

        # Add model parameters
        for key, value in model_params.items():
            if value is not None:
                args.extend([f"--model_{key}", str(value)])

        # Add eval parameters
        for key, value in eval_params.items():
            if value is not None:
                args.extend([f"--eval_{key}", str(value)])

        # Create job specification
        job = run_v1.Job()
        job.template.template.containers = [run_v1.Container()]
        container = job.template.template.containers[0]
        container.image = self.image_uri
        container.command = self.training_command
        container.args = args

        # Set resource requirements
        job.template.template.containers[0].resources.limits = {
            "cpu": "4000m",
            "memory": "16Gi",
            "nvidia.com/gpu": "1"
        }

        # Set timeout (10 hours)
        job.template.template.timeout.seconds = 36000

        return job

    def start_training_job(self, message_data: Dict[str, Any]) -> str:
        """Start a training job in Cloud Run."""
        try:
            job = self.create_job_request(message_data)

            # Generate unique job name
            import uuid
            custom_name = message_data.get("custom_name", "training_run")
            job_id = f"{custom_name}-{uuid.uuid4().hex[:8]}"

            # Create the job
            request = run_v1.CreateJobRequest(
                parent=f"projects/{self.project_id}/locations/{self.region}",
                job=job,
                job_id=job_id
            )

            operation = self.run_client.create_job(request)
            response = operation.result()

            logger.info(f"Created training job: {response.name}")
            return job_id

        except GoogleAPIError as e:
            if "already exists" in str(e):
                logger.warning(f"Job {job_id} already exists, this might be a retry")
                return job_id
            logger.error(f"Failed to create training job: {e}")
            raise

    def cancel_training_job(self, job_name: str) -> None:
        """Cancel a running training job."""
        try:
            request = run_v1.DeleteJobRequest(
                name=f"projects/{self.project_id}/locations/{self.region}/jobs/{job_name}"
            )

            operation = self.run_client.delete_job(request)
            operation.result()

            logger.info(f"Cancelled training job: {job_name}")

        except GoogleAPIError as e:
            logger.error(f"Failed to cancel training job {job_name}: {e}")
            raise
    def process_message(self, message: PubsubMessage) -> None:
        """Process a Pub/Sub message."""
        try:
            # Parse message
            message_data = self.parse_message(message.data)
            action = message_data.get("action")
            action = message_data.get("action")

            logger.info(f"Processing message with action: {action}")

            if action == "create":
                job_id = self.start_training_job(message_data)
                logger.info(f"Started training job: {job_id}")

            elif action == "cancel":
                job_id = message_data.get("job_id")
                if not job_id:
                    logger.error("Cancel action requires 'job_id' field")
                    return

                self.cancel_training_job(job_id)
                logger.info(f"Cancelled training job: {job_id}")

            else:
                logger.warning(f"Unknown action: {action}")

            # Acknowledge the message immediately (as per requirements)
            message.ack()

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Still acknowledge to prevent infinite retries
            message.ack()

    def process_message_data(self, message_data: Dict[str, Any]) -> None:
        """Process a message data dictionary (used by Cloud Functions)."""
        try:
            # Validate required fields
            if "action" not in message_data:
                raise ValueError("Message must contain 'action' field")

            action = message_data.get("action")
            logger.info(f"Processing message with action: {action}")

            if action == "create":
                job_id = self.start_training_job(message_data)
                logger.info(f"Started training job: {job_id}")

            elif action == "cancel":
                job_id = message_data.get("job_id")
                if not job_id:
                    logger.error("Cancel action requires 'job_id' field")
                    return

                self.cancel_training_job(job_id)
                logger.info(f"Cancelled training job: {job_id}")

            else:
                logger.warning(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise

    def run(self) -> None:
        """Run the Pub/Sub subscriber."""
        logger.info(f"Starting Pub/Sub subscriber for topic: {self.topic_name}")

        def callback(message) -> None:
            self.process_message(message)

        # Subscribe to the topic
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path, callback=callback
        )

        logger.info(f"Listening for messages on {self.subscription_path}...")

        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Subscriber stopped")


def process_pubsub_message(event, context):
    """Cloud Functions entry point for processing Pub/Sub messages."""
    import base64
    import json
    import logging
    import os

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Decode the Pub/Sub message
        if 'data' in event:
            message_data = base64.b64decode(event['data']).decode('utf-8')
            data = json.loads(message_data)
        else:
            logger.error("No data in Pub/Sub message")
            return

        # Create proxy instance and process message
        proxy = TrainingJobProxy()
        proxy.process_message_data(data)

        logger.info("Successfully processed Pub/Sub message")

    except Exception as e:
        logger.error(f"Error processing Pub/Sub message: {e}")
        raise  # Re-raise to mark function as failed


def process_message_data(self, message_data: Dict[str, Any]) -> None:
    """Process a message data dictionary (used by Cloud Functions)."""
    try:
        # Validate required fields
        if "action" not in message_data:
            raise ValueError("Message must contain 'action' field")

        action = message_data.get("action")
        logger.info(f"Processing message with action: {action}")

        if action == "create":
            job_id = self.start_training_job(message_data)
            logger.info(f"Started training job: {job_id}")

        elif action == "cancel":
            job_id = message_data.get("job_id")
            if not job_id:
                logger.error("Cancel action requires 'job_id' field")
                return

            self.cancel_training_job(job_id)
            logger.info(f"Cancelled training job: {job_id}")

        else:
            logger.warning(f"Unknown action: {action}")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise
