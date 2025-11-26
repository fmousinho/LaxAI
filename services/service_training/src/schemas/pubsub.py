"""Pub/Sub message schemas for training jobs."""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, Literal, List, Union
from pydantic import BaseModel, Field
import json


class PubSubMessage(BaseModel):
    """Base Pub/Sub message."""
    action: str = Field(..., description="Action type")
    task_id: str = Field(..., description="Training job task ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_json(self) -> str:
        """Serialize to JSON string for Pub/Sub.
        
        Returns:
            JSON string ready for publishing
        """
        data = self.dict()
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data)


class AutoResumeMessage(PubSubMessage):
    """Auto-resume message for suspended jobs."""
    action: Literal["auto_resume"] = "auto_resume"
    tenant_id: str
    wandb_run_name: str
    training_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training config including weights parameter for checkpoint control"
    )
    eval_params: Dict[str, Any] = Field(default_factory=dict)
    dataset_address: Optional[Union[str, List[str]]] = None
    auto_resume_count: int = 0


class CancellationMessage(PubSubMessage):
    """Cancellation request message."""
    action: Literal["cancel"] = "cancel"
    reason: Optional[str] = None
