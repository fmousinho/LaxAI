"""
Training API request/response schemas for the training service.
Explicit Pydantic models that use config defaults directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from shared_libs.config.all_config import (
    training_config,
    model_config,
    evaluator_config,
    wandb_config,
)


class TrainingParams(BaseModel):
    """Training hyperparameters - mirrors training_config."""

    num_epochs: int = Field(default=training_config.num_epochs, description="Number of training epochs")
    batch_size: int = Field(default=training_config.batch_size, description="Batch size for training")
    num_workers: int = Field(default=training_config.num_workers, description="Number of DataLoader workers")
    learning_rate: float = Field(default=training_config.learning_rate, description="Learning rate for optimizer")
    margin: float = Field(default=training_config.margin, description="Margin for triplet loss")
    weight_decay: float = Field(default=training_config.weight_decay, description="Weight decay for optimizer")

    lr_scheduler_patience: int = Field(default=training_config.lr_scheduler_patience, description="LR scheduler patience")
    lr_scheduler_threshold: float = Field(default=training_config.lr_scheduler_threshold, description="LR scheduler threshold")
    lr_scheduler_min_lr: float = Field(default=training_config.lr_scheduler_min_lr, description="LR scheduler minimum LR")
    lr_scheduler_factor: float = Field(default=training_config.lr_scheduler_factor, description="LR scheduler factor")

    force_pretraining: bool = Field(default=training_config.force_pretraining, description="Force pretrained ResNet defaults")
    early_stopping_patience: int = Field(default=training_config.early_stopping_patience, description="Early stopping patience")
    min_images_per_player: int = Field(default=training_config.min_images_per_player, description="Minimum images per player")
    train_prefetch_factor: int = Field(default=training_config.prefetch_factor, description="Prefetch factor for training data loading")

    margin_decay_rate: float = Field(default=training_config.margin_decay_rate, description="Triplet margin decay rate")
    margin_change_threshold: float = Field(default=training_config.margin_change_threshold, description="Min change in margin to update")
    train_ratio: float = Field(default=training_config.train_ratio, description="Train split ratio vs validation")
    n_datasets_to_use: Optional[int] = Field(default=training_config.n_datasets_to_use, description="Number of datasets to use (optional)")
    dataset_address: Optional[str] = Field(default=training_config.dataset_address, description="Specific dataset GCS path (optional)")

    wandb_project: str = Field(default=wandb_config.project, description="W&B project name")


class ModelParams(BaseModel):
    """Model architecture parameters - mirrors model_config."""

    embedding_dim: int = Field(default=model_config.embedding_dim, description="Embedding dimension")
    dropout_rate: float = Field(default=model_config.dropout_rate, description="Dropout rate")
    input_height: int = Field(default=model_config.input_height, description="Input image height")
    input_width: int = Field(default=model_config.input_width, description="Input image width")
    model_class_module: str = Field(default=model_config.model_class_module, description="Module where the model class is defined")
    model_class: str = Field(default=model_config.model_class_str, description="Model class name")


class EvalParams(BaseModel):
    """Evaluation parameters - mirrors evaluator_config."""

    number_of_workers: int = Field(default=evaluator_config.number_of_workers, description="Eval DataLoader workers")
    emb_batch_size: int = Field(default=evaluator_config.emb_batch_size, description="Eval embedding batch size")
    eval_prefetch_factor: int = Field(default=evaluator_config.prefetch_factor, description="Eval DataLoader prefetch factor")


class TrainingRequest(BaseModel):
    """Training request model with explicit parameter validation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "custom_name": "example_training_run",
                "tenant_id": "",
                "resume_from_checkpoint": False,
                "training_params": {
                    "num_epochs": training_config.num_epochs,
                    "batch_size": training_config.batch_size,
                    "learning_rate": training_config.learning_rate,
                },
                "model_params": {
                    "embedding_dim": model_config.embedding_dim,
                    "dropout_rate": model_config.dropout_rate,
                },
                "eval_params": {
                    "number_of_workers": evaluator_config.number_of_workers,
                    "emb_batch_size": evaluator_config.emb_batch_size,
                },
            }
        }
    )

    custom_name: str = Field(default="training_run", description="Custom name for the training run")
    tenant_id: str = Field(default="t", description="Tenant identifier for the training job")
    resume_from_checkpoint: bool = Field(default=True, description="Whether to resume from checkpoint if available")

    training_params: Optional[TrainingParams] = Field(default=None, description="Training-specific parameters")
    model_params: Optional[ModelParams] = Field(default=None, description="Model architecture parameters")
    eval_params: Optional[EvalParams] = Field(default=None, description="Evaluation parameters")


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
