"""
Cloud Run Job Worker for executing training jobs.

This worker receives Pub/Sub messages with task IDs, retrieves job details
from Firestore, executes training using the existing training service,
and updates progress in Firestore.

Designed to run as a Cloud Run Job with GPU support and long timeouts.
"""
import os
import json
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import base64

import config.logging_config

from utils.env_secrets import setup_environment_secrets
setup_environment_secrets()

# Configure logging

logger = logging.getLogger(__name__)


from google.cloud import pubsub_v1
from google.cloud import logging as cloud_logging
from concurrent.futures import ThreadPoolExecutor


# Import our custom modules
try:
    # Import cloud modules (same directory in deployment)
    from cloud.firestore_client import get_firestore_client, JobStatus
    
    # Import training modules from the main application
    from services.training_service import validate_training_params, _run_training_task
    from api.v1.schemas.training import TrainingRequest
        
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    # For deployment, these should be available
    raise

# Environment variables
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
SUBSCRIPTION_NAME = os.getenv('TRAINING_JOBS_SUBSCRIPTION', 'training-jobs-sub')
FIRESTORE_ENABLED = os.getenv('FIRESTORE_ENABLED', 'true').lower() == 'true'
WORKER_TIMEOUT = int(os.getenv('WORKER_TIMEOUT', '36000'))  # 10 hours in seconds
MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', '1'))  # GPU constraint

# Global state
_firestore_client = None
_pubsub_subscriber = None
_shutdown_requested = False


def get_firestore_client_instance():
    """Lazy initialization of Firestore client."""
    global _firestore_client
    if _firestore_client is None and FIRESTORE_ENABLED:
        _firestore_client = get_firestore_client(PROJECT_ID)
    return _firestore_client


def get_pubsub_subscriber():
    """Lazy initialization of Pub/Sub subscriber."""
    global _pubsub_subscriber
    if _pubsub_subscriber is None and pubsub_v1:
        _pubsub_subscriber = pubsub_v1.SubscriberClient()
    return _pubsub_subscriber


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    _shutdown_requested = True


def update_job_progress(firestore_client, task_id: str, progress: Dict[str, Any], status: Optional[str] = None):
    """Update job progress in Firestore."""
    try:
        update_data = {
            "progress": progress,
            "updated_at": datetime.utcnow()
        }
        
        if status:
            update_data["status"] = status
            if status == JobStatus.RUNNING.value:
                update_data["started_at"] = datetime.utcnow()
            elif status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
                update_data["completed_at"] = datetime.utcnow()
        
        firestore_client.update_job(task_id, update_data)
        logger.info(f"Updated job {task_id} progress: {progress}")
        
    except Exception as e:
        logger.error(f"Failed to update job progress for {task_id}: {e}")


def execute_training_job(task_id: str, job_data: Dict[str, Any]) -> bool:
    """
    Execute the actual training job.
    
    Args:
        task_id: Job identifier
        job_data: Job details from Firestore
        
    Returns:
        True if successful, False otherwise
    """
    firestore_client = get_firestore_client_instance()
    
    try:
        # Parse the training request
        request_payload = job_data.get("request_payload", {})
        training_request = TrainingRequest(**request_payload)
        
        logger.info(f"Starting training job {task_id} for tenant {training_request.tenant_id}")
        
        # Update status to RUNNING
        if firestore_client:
            update_job_progress(firestore_client, task_id, {"message": "Training started"}, JobStatus.RUNNING.value)
        
        # Progress callback to update Firestore during training
        def progress_callback(epoch: int, total_epochs: int, loss: float, metrics: Dict[str, float]):
            """Callback to report training progress."""
            if _shutdown_requested:
                logger.info("Shutdown requested, stopping training...")
                return False  # Signal to stop training
            
            progress_data = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "loss": loss,
                "metrics": metrics,
                "progress_percent": (epoch / total_epochs) * 100,
                "message": f"Training epoch {epoch}/{total_epochs}, loss: {loss:.4f}"
            }
            
            if firestore_client:
                update_job_progress(firestore_client, task_id, progress_data)
            
            return True  # Continue training
        
        # Execute the training using existing service
        # Note: We'll need to modify train_model to accept progress_callback
        logger.info(f"Executing training with params: {training_request.training_params}")
        
        # For now, simulate the training call - you'll need to modify your train_model function
        # to accept and use the progress_callback
        try:
            result = train_model(
                tenant_id=training_request.tenant_id,
                training_params=training_request.training_params,
                model_params=training_request.model_params,
                custom_name=training_request.custom_name,
                wandb_tags=training_request.wandb_tags,
                progress_callback=progress_callback
            )
            
            # Temporary simulation for demonstration
            logger.info("Simulating training execution...")
            total_epochs = getattr(training_request, 'training_params', {}).get("epochs", 10) if training_request.training_params else 10
            
            for epoch in range(1, total_epochs + 1):
                if _shutdown_requested:
                    logger.info("Training interrupted by shutdown signal")
                    if firestore_client:
                        update_job_progress(
                            firestore_client, 
                            task_id, 
                            {"message": "Training cancelled"}, 
                            JobStatus.CANCELLED.value
                        )
                    return False
                
                # Simulate training progress
                fake_loss = 1.0 - (epoch / total_epochs) * 0.8
                fake_metrics = {"accuracy": epoch / total_epochs * 0.95}
                
                success = progress_callback(epoch, total_epochs, fake_loss, fake_metrics)
                if not success:
                    break
                
                time.sleep(2)  # Simulate training time
            
            # Simulate successful completion
            result = {
                "model_path": f"/tmp/models/{task_id}/final_model.pth",
                "metrics": {"final_accuracy": 0.95, "final_loss": 0.2},
                "artifacts": {
                    "wandb_run_id": f"run_{task_id[:8]}",
                    "model_registry_id": f"model_{task_id[:8]}"
                }
            }
            
        except Exception as training_error:
            logger.error(f"Training execution failed for {task_id}: {training_error}")
            if firestore_client:
                update_job_progress(
                    firestore_client,
                    task_id,
                    {
                        "message": f"Training failed: {training_error}",
                        "error_details": str(training_error)
                    },
                    JobStatus.FAILED.value
                )
            return False
        
        # Training completed successfully
        if firestore_client:
            final_progress = {
                "message": "Training completed successfully",
                "results": result,
                "progress_percent": 100
            }
            update_job_progress(firestore_client, task_id, final_progress, JobStatus.COMPLETED.value)
        
        logger.info(f"Training job {task_id} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to execute training job {task_id}: {e}")
        
        if firestore_client:
            update_job_progress(
                firestore_client,
                task_id,
                {
                    "message": f"Job execution failed: {e}",
                    "error_details": str(e)
                },
                JobStatus.FAILED.value
            )
        
        return False


def process_pubsub_message(message):
    """
    Process a single Pub/Sub message containing a task ID.
    
    Args:
        message: Pub/Sub message object
    """
    try:
        # Decode message data
        message_data = json.loads(message.data.decode('utf-8'))
        task_id = message_data.get("task_id")
        
        if not task_id:
            logger.error("Message missing task_id")
            message.ack()
            return
        
        logger.info(f"Processing training job: {task_id}")
        
        # Retrieve job details from Firestore
        firestore_client = get_firestore_client_instance()
        if not firestore_client:
            logger.error("Firestore client not available")
            message.nack()
            return
        
        job_data = firestore_client.get_job(task_id)
        if not job_data:
            logger.error(f"Job {task_id} not found in Firestore")
            message.ack()  # Don't retry if job doesn't exist
            return
        
        # Check if job is already processed
        current_status = job_data.get("status")
        if current_status in [JobStatus.RUNNING.value, JobStatus.COMPLETED.value]:
            logger.info(f"Job {task_id} already processed (status: {current_status})")
            message.ack()
            return
        
        # Execute the training job
        success = execute_training_job(task_id, job_data)
        
        if success:
            logger.info(f"Successfully processed job {task_id}")
            message.ack()
        else:
            logger.error(f"Failed to process job {task_id}")
            # Don't retry failed jobs to avoid infinite loops
            message.ack()
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        message.nack()


def run_worker():
    """
    Main worker loop to process Pub/Sub messages.
    """
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting Cloud Run Job Worker...")
    logger.info(f"Project ID: {PROJECT_ID}")
    logger.info(f"Subscription: {SUBSCRIPTION_NAME}")
    logger.info(f"Max concurrent jobs: {MAX_CONCURRENT_JOBS}")
    logger.info(f"Worker timeout: {WORKER_TIMEOUT} seconds")
    
    subscriber = get_pubsub_subscriber()
    if not subscriber:
        logger.error("Pub/Sub subscriber not available")
        return

    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)

    # Flow control settings for GPU constraints
    if pubsub_v1:
        flow_control = pubsub_v1.types.FlowControl(max_messages=MAX_CONCURRENT_JOBS)
    else:
        logger.error("Pub/Sub library not available")
        return    logger.info(f"Listening for messages on {subscription_path}")

    # Use ThreadPoolExecutor for concurrent message processing
    if not ThreadPoolExecutor:
        logger.error("ThreadPoolExecutor not available")
        return
        
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        try:
            # Pull messages with flow control
            streaming_pull_future = subscriber.subscribe(
                subscription_path,
                callback=lambda message: executor.submit(process_pubsub_message, message),
                flow_control=flow_control
            )
            
            logger.info("Worker is ready to process messages...")
            
            # Keep the worker running until shutdown
            start_time = time.time()
            while not _shutdown_requested:
                elapsed = time.time() - start_time
                
                # Check for timeout (Cloud Run Job timeout)
                if elapsed > WORKER_TIMEOUT:
                    logger.info(f"Worker timeout reached ({WORKER_TIMEOUT}s), shutting down...")
                    break
                
                time.sleep(1)
            
            # Graceful shutdown
            logger.info("Shutting down worker...")
            streaming_pull_future.cancel()
            streaming_pull_future.result()  # Block until shutdown is complete
            
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            logger.info("Worker shutdown complete")


if __name__ == "__main__":
    # Check for local development mode
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        logger.info("Running in local development mode")
        # Mock a single job for testing
        mock_task_id = "test-job-123"
        mock_job_data = {
            "request_payload": {
                "tenant_id": "test-tenant",
                "training_params": {"epochs": 3, "batch_size": 16},
                "model_params": {"architecture": "yolo"},
                "custom_name": "local-test",
                "wandb_tags": ["local", "test"]
            }
        }
        
        # Test job execution
        success = execute_training_job(mock_task_id, mock_job_data)
        print(f"Local test result: {'SUCCESS' if success else 'FAILED'}")
    else:
        # Normal worker mode
        run_worker()
