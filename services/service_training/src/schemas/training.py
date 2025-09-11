from __future__ import annotations

from typing import Any, Dict, List, Optional

from parameter_registry import parameter_registry
from pydantic import BaseModel, ConfigDict, Field


def generate_example_from_config() -> Dict[str, Any]:
    """Generate example values from the training configuration."""
    
    # Get example values from the configuration
    example_data = {
        "custom_name": "example_training_run",
        "tenant_id": "",
        "resume_from_checkpoint": False
    }
    
    # Add training parameters with default values from parameter registry
    training_params = {}
    for param_name, param in parameter_registry.get_training_parameters().items():
        # Use the parameter's default value or a sensible example
        if param_name == "n_datasets_to_use":
            training_params[param_name] = 2
        elif param_name == "learning_rate":
            training_params[param_name] = 0.001
        elif param_name == "batch_size":
            training_params[param_name] = 32
        elif param_name == "epochs":
            training_params[param_name] = 100
        else:
            # Use a generic default based on parameter type
            if param.type.name == "INT":
                training_params[param_name] = 1
            elif param.type.name == "FLOAT":
                training_params[param_name] = 0.1
            elif param.type.name == "BOOL":
                training_params[param_name] = False
            elif param.type.name == "STR":
                training_params[param_name] = "default_value"
            elif param.type.name == "LIST_STR":
                training_params[param_name] = ["example"]
    example_data["training_params"] = training_params
    
    # Add model parameters with default values
    model_params = {}
    for param_name, param in parameter_registry.get_model_parameters().items():
        if param.type.name == "INT":
            model_params[param_name] = 512
        elif param.type.name == "FLOAT":
            model_params[param_name] = 0.1
        elif param.type.name == "BOOL":
            model_params[param_name] = True
        elif param.type.name == "STR":
            model_params[param_name] = "default"
        elif param.type.name == "LIST_STR":
            model_params[param_name] = ["example"]
    example_data["model_params"] = model_params
    
    # Add eval parameters with default values
    eval_params = {}
    try:
        for param_name, param in parameter_registry.eval_parameters.items():
            if param.type.name == "INT":
                eval_params[param_name] = 10
            elif param.type.name == "FLOAT":
                eval_params[param_name] = 0.5
            elif param.type.name == "BOOL":
                eval_params[param_name] = True
            elif param.type.name == "STR":
                eval_params[param_name] = "example"
            elif param.type.name == "LIST_STR":
                eval_params[param_name] = ["example"]
    except AttributeError:
        # If eval_parameters not available, use empty dict
        pass
    example_data["eval_params"] = eval_params
    
    return example_data


# Generate the example for use in model configuration
EXAMPLE_REQUEST = generate_example_from_config()


# Generate dynamic Pydantic models from parameter registry
TrainingParamsModel = None
ModelParamsModel = None
EvalParamsModel = None

try:
    from pydantic import create_model
    TrainingParamsModel = create_model(
        "TrainingParams",
        **parameter_registry.generate_pydantic_fields_for_training()
    )
    ModelParamsModel = create_model(
        "ModelParams", 
        **parameter_registry.generate_pydantic_fields_for_model()
    )
    EvalParamsModel = create_model(
        "EvalParams",
        **parameter_registry.generate_pydantic_fields_for_eval()
    )
except Exception:
    # Fallback if dynamic model creation fails
    pass


class TrainingRequest(BaseModel):
    """Training request model with dynamic parameter validation."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": EXAMPLE_REQUEST
        }
    )
    
    custom_name: str = Field(
        default="training_run",
        description="Custom name for the training run"
    )
    tenant_id: str = Field(
        default="t",
        description="Tenant identifier for the training job"
    )
    resume_from_checkpoint: bool = Field(
        default=True,
        description="Whether to resume from checkpoint if available"
    )
    
    # Use proper nested models for API schema expansion
    training_params: Optional[TrainingParamsModel] = Field(
        default_factory=lambda: TrainingParamsModel() if TrainingParamsModel else None,
        description="Training-specific parameters"
    )
    model_params: Optional[ModelParamsModel] = Field(
        default_factory=lambda: ModelParamsModel() if ModelParamsModel else None,
        description="Model architecture parameters"  
    )
    eval_params: Optional[EvalParamsModel] = Field(
        default_factory=lambda: EvalParamsModel() if EvalParamsModel else None,
        description="Evaluation parameters"
    )


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
