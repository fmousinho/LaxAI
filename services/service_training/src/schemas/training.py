"""
Training API request/response schemas for the training service.
Explicit Pydantic models that use config defaults directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal

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

    num_workers: int = Field(default=training_config.num_workers, description="Number of DataLoader workers")
    batch_size: int = Field(default=training_config.batch_size, description="Batch size for training")
    prefetch_factor: int = Field(default=training_config.prefetch_factor, description="Prefetch factor for training data loading")
    
    lr_initial: float = Field(default=training_config.learning_rate, description="Learning rate for optimizer")
    lr_scheduler_patience: int = Field(default=training_config.lr_scheduler_patience, description="LR scheduler patience")
    lr_scheduler_factor: float = Field(default=training_config.lr_scheduler_factor, description="LR scheduler factor")
    weight_decay: float = Field(default=training_config.weight_decay, description="Weight decay for optimizer")

    early_stopping_patience: int = Field(default=training_config.early_stopping_patience or 10, description="Early stopping patience")

    dataset_address: Union[str, List[str]] = Field(..., description="Specific dataset GCS path (string or list of str)")

    margin: float = Field(default=training_config.margin, description="Margin for triplet loss")
    weights: Literal["checkpoint", "latest", "reset"] = Field(
        default="checkpoint", 
        description="Use 'checkpoint' to resume from wandb, 'latest' for previous saved model, 'reset' to use ResNet defaults"
        )


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
    batch_size: int = Field(default=evaluator_config.eval_batch_size, description="Eval embedding batch size")
    prefetch_factor: int = Field(default=evaluator_config.prefetch_factor, description="Eval DataLoader prefetch factor")


class TrainingRequest(BaseModel):
    """Training request model with explicit parameter validation."""

    wandb_run_name: str = Field(default="training_run", description="Name used for WandB reporting")
    tenant_id: str = Field(default="t", description="Tenant identifier for the training job")

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
