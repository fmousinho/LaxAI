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
    learning_rate: Optional[float] = Field(None, description="Learning rate for optimizer")
    margin: Optional[float] = Field(None, description="Margin for triplet loss")
    weight_decay: Optional[float] = Field(None, description="Weight decay for optimizer")
    lr_scheduler_patience: Optional[int] = Field(None, description="Patience for learning rate scheduler")
    lr_scheduler_threshold: Optional[float] = Field(None, description="Threshold for learning rate scheduler")
    lr_scheduler_min_lr: Optional[float] = Field(None, description="Minimum learning rate for scheduler")
    lr_scheduler_factor: Optional[float] = Field(None, description="Factor for LR scheduler")
    force_pretraining: Optional[bool] = Field(None, description="Force pretrained ResNet defaults")
    early_stopping_patience: Optional[int] = Field(None, description="Early stopping patience")
    min_images_per_player: Optional[int] = Field(None, description="Minimum images required for a player")
    prefetch_factor: Optional[int] = Field(None, description="Prefetch factor for data loading")
    margin_decay_rate: Optional[float] = Field(None, description="Rate at which to decay the triplet loss margin")
    margin_change_threshold: Optional[float] = Field(None, description="Minimum change in margin to trigger an update")
    train_ratio: Optional[float] = Field(None, description="Percentage of dataset samples for training")
    n_datasets_to_use: Optional[int] = Field(None, description="Number of datasets to use for training")
    wandb_project: Optional[str] = Field(None, description="WandB project name")
    dataset_address: Optional[str] = Field(None, description="Address of the dataset to use")

    # Classification-head controls
    use_classification_head: Optional[bool] = Field(
        None, description="Enable auxiliary classification head for early epochs"
    )
    classification_epochs: Optional[int] = Field(
        None, description="Number of epochs to blend in classification loss"
    )
    classification_weight_start: Optional[float] = Field(
        None, description="Initial weight for classification loss (decays to 0)"
    )


class ModelParams(KebabModel):
    """Model architecture parameters."""

    embedding_dim: Optional[int] = Field(None, description="Embedding dimension")
    dropout_rate: Optional[float] = Field(None, description="Dropout rate")
    input_height: Optional[int] = Field(None, description="Required height for input images")
    input_width: Optional[int] = Field(None, description="Required width for input images")
    model_class_module: Optional[str] = Field(None, description="Module where the model class is defined")
    model_class: Optional[str] = Field(None, description="Name of the model class")


class EvalParams(KebabModel):
    """Evaluation parameters."""

    number_of_workers: Optional[int] = Field(None, description="Number of workers for DataLoader")
    emb_batch_size: Optional[int] = Field(None, description="Batch size for embedding generation")
    prefetch_factor: Optional[int] = Field(None, description="Number of batches to prefetch")


class TrainingRequest(KebabModel):
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
    dataset_address: Optional[str] = Field(
        default=None,
        description="Address of the dataset to be used for training"
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
                "custom_name": "my_training_run",
                "tenant_id": "",
                "resume_from_checkpoint": True,
                "dataset_address": "gs://my-bucket/datasets/dataset1",
                "training_params": {
                    "num_epochs": 100,
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "margin": 0.5,
                    "weight_decay": 0.0001,
                    "early_stopping_patience": 10,
                    "train_ratio": 0.8,
                    "prefetch_factor": 2,
                    "n_datasets_to_use": 10,
                    "use_classification_head": true,
                    "classification_epochs": 20,
                    "classification_weight_start": 1.0,
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
                    "emb_batch_size": 64,
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
