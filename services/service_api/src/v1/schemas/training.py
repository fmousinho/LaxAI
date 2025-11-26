"""
Training API schemas for LaxAI API Service.

Complete schemas for API request/response validation with specific parameter fields.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


def _to_kebab(s: str) -> str:
    """Convert snake_case field names to kebab-case for JSON aliases."""
    return s.replace("_", "-")


class KebabModel(BaseModel):
    """Base model that accepts kebab-case aliases for all snake_case fields."""

    model_config = ConfigDict(
        populate_by_name=True,  # allow snake_case as well
        alias_generator=_to_kebab,  # accept kebab-case in JSON
    )


class TrainingParams(KebabModel):
    """Training-specific parameters."""

    num_epochs: Optional[int] = Field(None, description="Number of training epochs")
    batch_size: Optional[int] = Field(None, description="Batch size for training")
    num_workers: Optional[int] = Field(None, description="Number of DataLoader workers")
    prefetch_factor: Optional[int] = Field(None, description="Prefetch factor for training data loading")
    
    lr_initial: Optional[float] = Field(None, description="Initial learning rate for optimizer")
    lr_scheduler_patience: Optional[int] = Field(None, description="LR scheduler patience")
    lr_scheduler_factor: Optional[float] = Field(None, description="LR scheduler factor")
    
    early_stopping_patience: Optional[int] = Field(None, description="Early stopping patience")
    
    dataset_address: Optional[str] = Field(None, description="Specific dataset GCS path")
    
    margin: Optional[float] = Field(None, description="Margin for triplet loss")
    weights: Optional[str] = Field(
        None,
        description="Use 'checkpoint' to resume from wandb, 'latest' for previous saved model, 'reset' to use ResNet defaults"
    )


class ModelParams(KebabModel):
    """Model architecture parameters."""

    embedding_dim: Optional[int] = Field(None, description="Embedding dimension")
    dropout_rate: Optional[float] = Field(None, description="Dropout rate")
    input_height: Optional[int] = Field(None, description="Input image height")
    input_width: Optional[int] = Field(None, description="Input image width")
    model_class_module: Optional[str] = Field(None, description="Module where the model class is defined")
    model_class: Optional[str] = Field(None, description="Model class name")


class EvalParams(KebabModel):
    """Evaluation parameters."""

    number_of_workers: Optional[int] = Field(None, description="Eval DataLoader workers")
    batch_size: Optional[int] = Field(None, description="Eval embedding batch size")
    prefetch_factor: Optional[int] = Field(None, description="Eval DataLoader prefetch factor")


class TrainingRequest(KebabModel):
    """Training request model for API services."""

    wandb_run_name: str = Field(
        default="training_run",
        description="Name used for WandB reporting"
    )
    tenant_id: str = Field(
        default="t",
        description="Tenant identifier for the training job"
    )
    training_params: Optional[TrainingParams] = Field(
        default=None,
        description="Training-specific parameters"
    )
    model_params: Optional[ModelParams] = Field(
        default=None,
        description="Model architecture parameters"
    )
    eval_params: Optional[EvalParams] = Field(
        default=None,
        description="Evaluation parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "wandb_run_name": "my_training_run",
                "tenant_id": "t",
                "training_params": {
                    "num_epochs": 100,
                    "batch_size": 128,
                    "num_workers": 4,
                    "prefetch_factor": 2,
                    "lr_initial": 0.001,
                    "lr_scheduler_patience": 10,
                    "lr_scheduler_factor": 0.5,
                    "early_stopping_patience": 15,
                    "dataset_address": "gs://laxai-datasets/training/dataset1",
                    "margin": 0.5,
                    "weights": "checkpoint"
                },
                "model_params": {
                    "embedding_dim": 512,
                    "dropout_rate": 0.1,
                    "input_height": 224,
                    "input_width": 224,
                    "model_class_module": "siamesenet_dino",
                    "model_class": "SiameseNet"
                },
                "eval_params": {
                    "number_of_workers": 4,
                    "batch_size": 64,
                    "prefetch_factor": 2
                }
            }
        }


class TrainingResponse(KebabModel):
    """Training response model."""

    task_id: str = Field(..., description="Unique identifier for the training task")
    status: str = Field(..., description="Current status of the training task")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="Task creation timestamp")


class TrainingStatus(KebabModel):
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


class ErrorResponse(KebabModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
