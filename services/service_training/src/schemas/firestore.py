"""Firestore document schemas for training jobs."""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

from ..state_machine import TrainingJobState


class TrainingJobDocument(BaseModel):
    """Complete Firestore document schema for training jobs.
    
    This is the single source of truth for job state.
    """
    
    # Identity
    task_id: str = Field(..., description="Unique task identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # State
    status: TrainingJobState = Field(..., description="Current job state")
    
    # Configuration
    wandb_run_name: str = Field(..., description="WandB run name")
    training_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training config including weights parameter for checkpoint control"
    )
    eval_params: Dict[str, Any] = Field(default_factory=dict)
    dataset_address: Optional[Union[str, List[str]]] = Field(
        None,
        description="Can be single path or list of paths"
    )
    
    # Execution metadata
    run_guids: List[str] = Field(default_factory=list)
    auto_resume_count: int = Field(default=0)
    
    # Progress tracking
    progress: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Overall progress percentage"
    )
    total_epochs: Optional[int] = Field(
        None,
        description="Total epochs configured (from training_params)"
    )
    
    # Metrics - high-level summary only
    # Detailed metrics are tracked in the metrics module
    metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Summary metrics snapshot"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Error handling
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list, max_items=100)
    
    def to_firestore(self) -> Dict[str, Any]:
        """Convert to Firestore format with ISO timestamps.
        
        Note: The 'Z' suffix explicitly indicates UTC timezone in ISO 8601 format.
        While Python's isoformat() includes timezone info, the 'Z' is the standard
        way to denote UTC and is widely recognized by APIs and databases.
        
        Returns:
            Dictionary ready for Firestore storage
        """
        data = self.dict()
        data['created_at'] = self.created_at.isoformat() + 'Z'
        data['updated_at'] = self.updated_at.isoformat() + 'Z'
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat() + 'Z'
        data['status'] = self.status.value  # Convert enum to string
        return data
    
    @classmethod
    def from_firestore(cls, data: Dict[str, Any]) -> "TrainingJobDocument":
        """Create from Firestore document.
        
        Args:
            data: Dictionary from Firestore
            
        Returns:
            TrainingJobDocument instance
        """
        # Convert ISO strings back to datetime
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'].rstrip('Z'))
        if 'updated_at' in data:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'].rstrip('Z'))
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'].rstrip('Z'))
        return cls(**data)
