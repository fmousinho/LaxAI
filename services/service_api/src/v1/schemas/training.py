"""
Training API schemas for LaxAI API Service.

Simplified schemas for API request/response validation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    """Training request model for API services."""

    custom_name: str = Field(
        default="training_run",
        description="Custom name for the training run"
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for the training job"
    )
    resume_from_checkpoint: bool = Field(
        default=True,
        description="Whether to resume from checkpoint if available"
    )
    training_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Training-specific parameters"
    )
    model_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model architecture parameters"
    )
    eval_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Evaluation parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "custom_name": "my_training_run",
                "tenant_id": "tenant1",
                "resume_from_checkpoint": True,
                "training_params": {
                    "batch_size": 128,
                    "num_epochs": 100,
                    "learning_rate": 0.001
                },
                "model_params": {
                    "embedding_dim": 512,
                    "dropout_rate": 0.1
                },
                "eval_params": {
                    "threshold": 0.7
                }
            }
        }


class TrainingResponse(BaseModel):
    """Training response model."""

    task_id: str = Field(..., description="Unique identifier for the training task")
    status: str = Field(..., description="Current status of the training task")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="Task creation timestamp")


class TrainingStatus(BaseModel):
    """Training status model."""

    task_id: str = Field(..., description="Unique identifier for the training task")
    status: str = Field(..., description="Current status of the training task")
    progress: Optional[float] = Field(None, description="Training progress percentage (0-100)")
    current_epoch: Optional[int] = Field(None, description="Current training epoch")
    total_epochs: Optional[int] = Field(None, description="Total training epochs")
    loss: Optional[float] = Field(None, description="Current training loss")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics")
    logs: Optional[List[str]] = Field(None, description="Recent training logs")
    updated_at: str = Field(..., description="Last update timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
