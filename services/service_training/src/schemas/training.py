from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

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
    
        # Add training parameters with default values from config only
    training_params = {}
    for param_name, param in parameter_registry.get_training_parameters().items():
        try:
            # Get the actual default value from config
            config_value = parameter_registry.get_config_value(param_name)
            # Include the parameter even if config_value is None (for optional parameters)
            training_params[param_name] = config_value
        except Exception as e:
            if param.required:
                raise ValueError(f"Failed to retrieve required training parameter '{param_name}' from config: {e}")
            # For optional parameters, include with None if config retrieval fails
            training_params[param_name] = None
    
    example_data["training_params"] = training_params
    
    # Add model parameters with default values from config only
    model_params = {}
    for param_name, param in parameter_registry.get_model_parameters().items():
        try:
            # Get the actual default value from config
            config_value = parameter_registry.get_config_value(param_name)
            # Include the parameter even if config_value is None (for optional parameters)
            model_params[param_name] = config_value
        except Exception as e:
            if param.required:
                raise ValueError(f"Failed to retrieve required model parameter '{param_name}' from config: {e}")
            # For optional parameters, include with None if config retrieval fails
            model_params[param_name] = None
    example_data["model_params"] = model_params
    
    # Add eval parameters with default values from config only
    eval_params = {}
    try:
        for param_name, param in parameter_registry.eval_parameters.items():
            try:
                # Get the actual default value from config
                config_value = parameter_registry.get_config_value(param_name)
                # Include the parameter even if config_value is None (for optional parameters)
                eval_params[param_name] = config_value
            except Exception as e:
                if param.required:
                    raise ValueError(f"Failed to retrieve required eval parameter '{param_name}' from config: {e}")
                # For optional parameters, include with None if config retrieval fails
                eval_params[param_name] = None
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
    from pydantic import ConfigDict, create_model

    # Create the models without trying to add config for now
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
        
except Exception as e:
    print(f"Warning: Dynamic model creation failed: {e}")
    # Fallback if dynamic model creation fails
    pass


class TrainingRequest(BaseModel):
    """Training request model with dynamic parameter validation."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "custom_name": "example_training_run",
                "tenant_id": "tenant1",
                "resume_from_checkpoint": False,
                "training_params": EXAMPLE_REQUEST.get("training_params", {}),
                "model_params": EXAMPLE_REQUEST.get("model_params", {}),
                "eval_params": EXAMPLE_REQUEST.get("eval_params", {})
            }
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
    
    # Use the actual dynamic models for proper schema generation
    training_params: Optional[Any] = Field(
        default=None,
        description="Training-specific parameters"
    )
    model_params: Optional[Any] = Field(
        default=None,
        description="Model architecture parameters"
    )
    eval_params: Optional[Any] = Field(
        default=None,
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
