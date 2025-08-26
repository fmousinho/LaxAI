"""
Cloud Function for orchestrating training jobs.

This function receives training requests, validates them using the existing
training service, stores job details in Firestore, and publishes task IDs
to Pub/Sub for processing by Cloud Run Jobs.
"""
import os
import json
import logging
from typing import Dict, Any, Tuple, Optional, Union
import functions_framework
from flask import Request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from google.cloud import pubsub_v1
    from google.cloud import logging as cloud_logging
except ImportError:
    logger.warning("Google Cloud libraries not available - running in local mode")
    pubsub_v1 = None
    cloud_logging = None

# Import our custom modules
from cloud.firestore_client import get_firestore_client, JobStatus
from services.training_service import validate_training_params
# Import TrainingRequest lazily to avoid heavy ML config dependencies
# from api.v1.schemas.training import TrainingRequest

# Environment variables
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
PUBSUB_TOPIC = os.getenv('TRAINING_JOBS_TOPIC', 'training-jobs')
FIRESTORE_ENABLED = os.getenv('FIRESTORE_ENABLED', 'true').lower() == 'true'

# Initialize clients (will be lazy-loaded)
_firestore_client = None
_pubsub_publisher = None


def get_firestore_client_instance():
    """Lazy initialization of Firestore client."""
    global _firestore_client
    if _firestore_client is None and FIRESTORE_ENABLED:
        _firestore_client = get_firestore_client(PROJECT_ID)
    return _firestore_client


def get_pubsub_publisher():
    """Lazy initialization of Pub/Sub publisher."""
    global _pubsub_publisher
    if _pubsub_publisher is None and pubsub_v1:
        _pubsub_publisher = pubsub_v1.PublisherClient()
    return _pubsub_publisher


def validate_request_payload(request_data: Dict[str, Any]) -> Tuple[bool, str, Optional[Any]]:
    """
    Validate the incoming training request.
    
    Args:
        request_data: Raw request payload
        
    Returns:
        Tuple of (is_valid, error_message, parsed_request)
    """
    try:
        # Import TrainingRequest here to avoid heavy ML config dependencies during module import
        from api.v1.schemas.training import TrainingRequest
        
        # Parse into TrainingRequest model for validation
        training_request = TrainingRequest(**request_data)
        
        # Use existing service validation for training_params
        try:
            validate_training_params(training_request.training_params)
        except Exception as validation_error:
            return False, f"Training params validation failed: {validation_error}", None
            
        return True, "", training_request
        
    except Exception as e:
        return False, f"Request validation failed: {e}", None


def publish_job_to_pubsub(task_id: str) -> bool:
    """
    Publish task ID to Pub/Sub topic for worker processing.
    
    Args:
        task_id: Job identifier to publish
        
    Returns:
        True if published successfully, False otherwise
    """
    try:
        publisher = get_pubsub_publisher()
        if not publisher:
            logger.warning("Pub/Sub publisher not available")
            return False
            
        topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
        
        # Message payload - just the task_id for the worker
        message_data = json.dumps({"task_id": task_id}).encode('utf-8')
        
        # Publish message
        future = publisher.publish(topic_path, message_data)
        message_id = future.result()  # Block until published
        
        logger.info(f"Published task {task_id} to Pub/Sub. Message ID: {message_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to publish task {task_id} to Pub/Sub: {e}")
        return False


@functions_framework.http
def orchestrate_training_job(request: Request):
    """
    Cloud Function entry point for training job orchestration.
    
    Accepts POST requests with training job parameters, validates them,
    stores in Firestore, and publishes to Pub/Sub for worker processing.
    
    Expected request format:
    {
        "tenant_id": "string",
        "training_params": {...},
        "model_params": {...},
        "custom_name": "string",
        "wandb_tags": [...]
    }
    
    Returns:
    {
        "task_id": "string",
        "status": "queued",
        "message": "Job queued successfully"
    }
    """
    # Enable CORS for browser requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Only accept POST requests
    if request.method != 'POST':
        return jsonify({
            "error": "Method not allowed",
            "message": "Only POST requests are supported"
        }), 405
    
    try:
        # Parse request JSON
        try:
            request_data = request.get_json(force=True)
            if not request_data:
                return jsonify({
                    "error": "Invalid request",
                    "message": "Request body must be valid JSON"
                }), 400
        except Exception as e:
            return jsonify({
                "error": "Invalid JSON",
                "message": f"Failed to parse request JSON: {e}"
            }), 400
        
        logger.info(f"Received training job request: {request_data.get('tenant_id', 'unknown')}")
        
        # Validate request using existing service validation
        is_valid, error_msg, training_request = validate_request_payload(request_data)
        if not is_valid:
            logger.warning(f"Request validation failed: {error_msg}")
            return jsonify({
                "error": "Validation failed",
                "message": error_msg
            }), 422
        
        # Generate unique task ID (reuse existing logic)
        import uuid
        task_id = str(uuid.uuid4())
        
        # Store job in Firestore
        firestore_client = get_firestore_client_instance()
        if firestore_client and training_request:
            try:
                job_metadata = {
                    "source": "cloud_function",
                    "function_name": "orchestrate_training_job"
                }
                
                firestore_client.create_job(
                    task_id=task_id,
                    request_payload=training_request.model_dump(),
                    metadata=job_metadata
                )
                logger.info(f"Stored job {task_id} in Firestore")
                
            except Exception as e:
                logger.error(f"Failed to store job in Firestore: {e}")
                return jsonify({
                    "error": "Storage failed",
                    "message": "Failed to store job details"
                }), 500
        else:
            logger.warning("Firestore client not available - job not persisted")
        
        # Publish to Pub/Sub for worker processing
        publish_success = publish_job_to_pubsub(task_id)
        if not publish_success:
            # If Pub/Sub fails, we should clean up Firestore and return error
            if firestore_client:
                try:
                    firestore_client.delete_job(task_id)
                except Exception:
                    pass  # Best effort cleanup
            
            return jsonify({
                "error": "Queue failed",
                "message": "Failed to queue job for processing"
            }), 500
        
        # Success response
        response_data = {
            "task_id": task_id,
            "status": "queued",
            "message": "Training job queued successfully"
        }
        
        logger.info(f"Successfully orchestrated training job: {task_id}")
        
        # Add CORS headers
        response = jsonify(response_data)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
        
    except Exception as e:
        logger.error(f"Unexpected error in orchestrate_training_job: {e}")
        return jsonify({
            "error": "Internal error",
            "message": "An unexpected error occurred"
        }), 500


@functions_framework.http 
def get_job_status(request: Request):
    """
    Cloud Function to retrieve job status from Firestore.
    
    Accepts GET requests with task_id parameter.
    Example: GET /get_job_status?task_id=abc-123
    
    Returns job status and details from Firestore.
    """
    # Enable CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    if request.method != 'GET':
        return jsonify({
            "error": "Method not allowed",
            "message": "Only GET requests are supported"
        }), 405
    
    try:
        # Get task_id from query parameters
        task_id = request.args.get('task_id')
        if not task_id:
            return jsonify({
                "error": "Missing parameter",
                "message": "task_id parameter is required"
            }), 400
        
        # Retrieve job from Firestore
        firestore_client = get_firestore_client_instance()
        if not firestore_client:
            return jsonify({
                "error": "Service unavailable", 
                "message": "Job status service is not available"
            }), 503
        
        job_data = firestore_client.get_job(task_id)
        if not job_data:
            return jsonify({
                "error": "Job not found",
                "message": f"No job found with task_id: {task_id}"
            }), 404
        
        # Format response (convert datetime objects to ISO strings)
        def format_datetime(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return obj
        
        # Clean up response data
        response_data = {
            "task_id": job_data["task_id"],
            "status": job_data["status"],
            "created_at": format_datetime(job_data.get("created_at")),
            "updated_at": format_datetime(job_data.get("updated_at")),
            "started_at": format_datetime(job_data.get("started_at")),
            "completed_at": format_datetime(job_data.get("completed_at")),
            "progress": job_data.get("progress", {}),
            "error_details": job_data.get("error_details")
        }
        
        response = jsonify(response_data)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving job status for {task_id}: {e}")
        return jsonify({
            "error": "Internal error",
            "message": "Failed to retrieve job status"
        }), 500


# Local development server (optional)
if __name__ == "__main__":
    try:
        import flask
        from werkzeug.serving import run_simple
    except ImportError:
        print("Flask/Werkzeug not available for local development server")
        print("Install with: pip install flask werkzeug")
        exit(1)
    
    app = flask.Flask(__name__)
    app.add_url_rule('/orchestrate', 'orchestrate', orchestrate_training_job, methods=['POST', 'OPTIONS'])
    app.add_url_rule('/status', 'status', get_job_status, methods=['GET', 'OPTIONS'])
    
    print("Starting local development server...")
    print("POST /orchestrate - Submit training jobs")
    print("GET /status?task_id=<id> - Check job status")
    run_simple('localhost', 8080, app, use_reloader=True, use_debugger=True)
