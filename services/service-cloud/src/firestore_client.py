"""
Firestore client for managing training job state.

This module provides utilities for creating, updating, and querying
training job documents in Firestore. Used by both the Cloud Function
orchestrator and Cloud Run Job worker.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

try:
    from google.cloud import firestore
    from google.cloud.firestore_v1 import Client
except ImportError:
    raise ImportError(
        "google-cloud-firestore is required. Install with: pip install google-cloud-firestore"
    )

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Training job status states."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING" 
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FirestoreJobClient:
    """
    Client for managing training job documents in Firestore.
    
    Jobs are stored in the 'training_jobs' collection with the following schema:
    {
        "task_id": str,           # Unique job identifier
        "status": str,            # JobStatus enum value
        "created_at": datetime,   # Job creation timestamp
        "updated_at": datetime,   # Last status update
        "started_at": datetime,   # When job execution began (optional)
        "completed_at": datetime, # When job finished (optional)
        "request_payload": dict,  # Original training request
        "progress": dict,         # Training progress metrics
        "error_details": dict,    # Error information if failed (optional)
        "metadata": dict,         # Additional job metadata
    }
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize Firestore client.
        
        Args:
            project_id: GCP project ID. If None, uses default from environment.
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError(
                "project_id must be provided or GOOGLE_CLOUD_PROJECT env var must be set"
            )
        
        self.client: Client = firestore.Client(project=self.project_id)
        self.collection_name = "training_jobs"
        self.jobs_collection = self.client.collection(self.collection_name)
        
        logger.info(f"Initialized FirestoreJobClient for project: {self.project_id}")
    
    def create_job(self, task_id: str, request_payload: Dict[str, Any], 
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new training job document.
        
        Args:
            task_id: Unique job identifier
            request_payload: Original training request data
            metadata: Additional job metadata
            
        Returns:
            Created job document
            
        Raises:
            Exception: If job creation fails
        """
        now = datetime.utcnow()
        
        job_doc = {
            "task_id": task_id,
            "status": JobStatus.QUEUED.value,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "request_payload": request_payload,
            "progress": {
                "current_epoch": None,
                "total_epochs": None,
                "current_loss": None,
                "best_loss": None,
                "message": "Job queued for execution"
            },
            "error_details": None,
            "metadata": metadata or {}
        }
        
        try:
            # Use task_id as document ID for easy retrieval
            doc_ref = self.jobs_collection.document(task_id)
            doc_ref.set(job_doc)
            
            logger.info(f"Created job document for task_id: {task_id}")
            return job_doc
            
        except Exception as e:
            logger.error(f"Failed to create job document for {task_id}: {e}")
            raise
    
    def get_job(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a job document by task_id.
        
        Args:
            task_id: Job identifier
            
        Returns:
            Job document dict or None if not found
        """
        try:
            doc_ref = self.jobs_collection.document(task_id)
            doc = doc_ref.get()
            
            if doc.exists:
                job_data = doc.to_dict()
                logger.debug(f"Retrieved job: {task_id}")
                return job_data
            else:
                logger.warning(f"Job not found: {task_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve job {task_id}: {e}")
            raise
    
    def update_job_status(self, task_id: str, status: JobStatus, 
                         progress: Optional[Dict[str, Any]] = None,
                         error_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update job status and optional progress/error information.
        
        Args:
            task_id: Job identifier
            status: New job status
            progress: Progress information to update
            error_details: Error details if status is FAILED
            
        Returns:
            True if update succeeded, False otherwise
        """
        try:
            doc_ref = self.jobs_collection.document(task_id)
            now = datetime.utcnow()
            
            update_data = {
                "status": status.value,
                "updated_at": now
            }
            
            # Set started_at when job begins running
            if status == JobStatus.RUNNING:
                update_data["started_at"] = now
            
            # Set completed_at when job finishes (success or failure)
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                update_data["completed_at"] = now
            
            # Update progress if provided
            if progress:
                update_data["progress"] = progress
            
            # Update error details if provided
            if error_details:
                update_data["error_details"] = error_details
            
            doc_ref.update(update_data)
            logger.info(f"Updated job {task_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update job {task_id} status: {e}")
            return False
    
    def list_jobs(self, status_filter: Optional[JobStatus] = None, 
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        List jobs, optionally filtered by status.
        
        Args:
            status_filter: Only return jobs with this status
            limit: Maximum number of jobs to return
            
        Returns:
            List of job documents
        """
        try:
            query = self.jobs_collection.order_by("created_at", direction=firestore.Query.DESCENDING)
            
            if status_filter:
                query = query.where("status", "==", status_filter.value)
            
            docs = query.limit(limit).get()
            jobs = [doc.to_dict() for doc in docs]
            
            logger.info(f"Listed {len(jobs)} jobs (filter: {status_filter})")
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise
    
    def delete_job(self, task_id: str) -> bool:
        """
        Delete a job document.
        
        Args:
            task_id: Job identifier
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            doc_ref = self.jobs_collection.document(task_id)
            doc_ref.delete()
            
            logger.info(f"Deleted job: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete job {task_id}: {e}")
            return False


def get_firestore_client(project_id: Optional[str] = None) -> FirestoreJobClient:
    """
    Factory function to create a FirestoreJobClient instance.
    
    Args:
        project_id: GCP project ID
        
    Returns:
        Configured FirestoreJobClient instance
    """
    return FirestoreJobClient(project_id=project_id)


# Example usage functions for testing
if __name__ == "__main__":
    import uuid
    
    # Example usage
    client = get_firestore_client()
    
    # Create a test job
    test_task_id = f"test_job_{uuid.uuid4().hex[:8]}"
    test_payload = {
        "tenant_id": "test_tenant",
        "training_params": {"num_epochs": 10, "batch_size": 32},
        "model_params": {}
    }
    
    # Create job
    job = client.create_job(test_task_id, test_payload, {"test": True})
    print(f"Created job: {job}")
    
    # Update status to running
    client.update_job_status(test_task_id, JobStatus.RUNNING, 
                           progress={"current_epoch": 1, "message": "Training started"})
    
    # Retrieve job
    retrieved_job = client.get_job(test_task_id)
    print(f"Retrieved job: {retrieved_job}")
    
    # List jobs
    jobs = client.list_jobs(limit=5)
    print(f"Found {len(jobs)} recent jobs")
    
    # Clean up test job
    client.delete_job(test_task_id)
    print(f"Deleted test job: {test_task_id}")
