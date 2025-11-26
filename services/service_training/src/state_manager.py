"""Centralized state manager for training jobs."""

from typing import Optional
from datetime import datetime, timezone
import logging

try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

from .schemas.firestore import TrainingJobDocument
from .state_machine import TrainingJobState, StateTransition

logger = logging.getLogger(__name__)


class TrainingJobStateManager:
    """Centralized manager for training job state in Firestore."""
    
    COLLECTION_NAME = "training_runs"
    
    def __init__(self, firestore_client: Optional['firestore.Client'] = None):
        """Initialize state manager.
        
        Args:
            firestore_client: Optional Firestore client. If None, creates new one.
        """
        if not FIRESTORE_AVAILABLE:
            logger.warning("Firestore not available, state manager disabled")
            self.client = None
            return
            
        try:
            self.client = firestore_client or firestore.Client()
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            self.client = None
    
    def create_job(self, job: TrainingJobDocument) -> bool:
        """Create a new training job in Firestore.
        
        Args:
            job: Training job document to create
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Firestore client not available, skipping job creation")
            return False
            
        try:
            doc_ref = self.client.collection(self.COLLECTION_NAME).document(job.task_id)
            doc_ref.set(job.to_firestore())
            logger.info(f"Created training job {job.task_id} with status {job.status}")
            return True
        except Exception as e:
            logger.error(f"Failed to create job {job.task_id}: {e}")
            return False
    
    def get_job(self, task_id: str) -> Optional[TrainingJobDocument]:
        """Retrieve a training job from Firestore.
        
        Args:
            task_id: Task ID to retrieve
            
        Returns:
            TrainingJobDocument if found, None otherwise
        """
        if not self.client:
            return None
            
        try:
            doc_ref = self.client.collection(self.COLLECTION_NAME).document(task_id)
            doc = doc_ref.get()
            if doc.exists:
                return TrainingJobDocument.from_firestore(doc.to_dict())
            return None
        except Exception as e:
            logger.error(f"Failed to get job {task_id}: {e}")
            return None
    
    def update_status(
        self, 
        task_id: str, 
        new_status: TrainingJobState, 
        error: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Update job status with transition validation.
        
        Args:
            task_id: Task ID to update
            new_status: New status to set
            error: Optional error message
            **kwargs: Additional fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning(f"Firestore client not available, skipping status update for {task_id}")
            return False
            
        # Get current job to validate transition
        job = self.get_job(task_id)
        if not job:
            logger.error(f"Job {task_id} not found")
            return False
        
        # Validate transition
        if not StateTransition.is_valid_transition(job.status, new_status):
            logger.error(
                f"Invalid state transition for {task_id}: "
                f"{job.status.value} → {new_status.value}"
            )
            return False
        
        # Update document
        try:
            update_data = {
                "status": new_status.value,
                "updated_at": datetime.now(timezone.utc).isoformat() + "Z"
            }
            
            if error:
                update_data["error"] = error
            
            # Add any additional fields
            update_data.update(kwargs)
            
            # Mark completion time for terminal states
            if StateTransition.is_terminal_state(new_status):
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat() + "Z"
            
            doc_ref = self.client.collection(self.COLLECTION_NAME).document(task_id)
            doc_ref.update(update_data)
            logger.info(f"Updated {task_id}: {job.status.value} → {new_status.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to update status for {task_id}: {e}")
            return False
    
    def update_progress(self, task_id: str, **kwargs) -> bool:
        """Update progress fields without changing status.
        
        Args:
            task_id: Task ID to update
            **kwargs: Fields to update (progress, metrics, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
            
        try:
            update_data = {
                "updated_at": datetime.now(timezone.utc).isoformat() + "Z",
                **kwargs
            }
            
            doc_ref = self.client.collection(self.COLLECTION_NAME).document(task_id)
            doc_ref.update(update_data)
            return True
        except Exception as e:
            logger.error(f"Failed to update progress for {task_id}: {e}")
            return False
