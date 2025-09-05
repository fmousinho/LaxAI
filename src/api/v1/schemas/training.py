from pydantic import BaseModel, Field, create_model, ConfigDict
from typing import Optional, Dict, Any, List
from fastapi import HTTPException
from config.parameter_registry import parameter_registry
from config.all_config import api_config

def create_training_config_model():
    """Dynamically create TrainingConfig model from parameter registry"""
    try:
        fields = parameter_registry.generate_pydantic_fields_for_training()
        model = create_model(
            'TrainingConfig',
            __doc__="Training configuration parameters for model training.",
            **fields
        )
        return model
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create training configuration from parameter registry: {str(e)}"
        )

def create_model_config_model():
    """Dynamically create ModelConfig model from parameter registry"""
    try:
        fields = parameter_registry.generate_pydantic_fields_for_model()
        model = create_model(
            'ModelConfig', 
            __doc__="Model configuration parameters. Do NOT use if you don't know what you are doing.",
            **fields
        )
        return model
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create model configuration from parameter registry: {str(e)}"
        )


TrainingConfig = create_training_config_model()
ModelConfig = create_model_config_model()

def create_eval_config_model():
    """Dynamically create EvalConfig model from parameter registry"""
    try:
        from config.parameter_registry import parameter_registry
        fields = parameter_registry.generate_pydantic_fields_for_eval()
        model = create_model(
            'EvalConfig',
            __doc__="Evaluation configuration parameters.",
            **fields
        )
        return model
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create eval configuration from parameter registry: {str(e)}"
        )

EvalConfig = create_eval_config_model()

class TrainingRequest(BaseModel):
    """Request schema for training endpoint"""
    tenant_id: str = Field(default="", description="Tenant identifier")
    verbose: bool = Field(default=api_config.verbose, description="Enable verbose logging")
    custom_name: str = Field(default="", description="Label used by WandB")
    resume_from_checkpoint: bool = Field(default=api_config.resume_from_checkpoint, description="Resume from checkpoint if available")
    wandb_tags: List[str] = Field(default=[], description="WandB tags for this run")
    # Number of discovered datasets to use for the run. This is a top-level
    # field (not part of the dynamic training config) so callers can control
    # dataset selection without embedding it in training_params.
    n_datasets_to_use: Optional[int] = Field(default=None, description="Limit number of discovered datasets to use for training")
    # Training and model params are generated dynamically at runtime from the
    # parameter registry. Use the generated Pydantic models so FastAPI/OpenAPI
    # renders them in the UI and request bodies are validated into model objects.
    training_params: Optional['TrainingConfig'] = None  #type: ignore[valid-type]
    model_params: Optional['ModelConfig'] = None. #type: ignore[valid-type]
    eval_params: Optional['EvalConfig'] = None #type: ignore[valid-type]

class TrainingResponse(BaseModel):
    """Response schema for training endpoint"""
    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Current status")
    message: str = Field(description="Status message")

class TrainingProgress(BaseModel):
    """Training progress tracking"""
    task_id: str
    status: str
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_loss: Optional[float] = None
    elapsed_time: Optional[float] = None
    estimated_time_remaining: Optional[float] = None

class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str = Field(description="Error details")
    error_type: str = Field(description="Type of error")

def get_training_example() -> Dict[str, Any]:
    """Get example training configuration with dynamic values from parameter registry"""
    try:
        example_config = {}
        
        # Get all parameter values from registry
        for param_name, param_def in parameter_registry.parameters.items():
            try:
                value = parameter_registry.get_config_value(param_name)
                if value is not None:
                    example_config[param_name] = value
            except Exception:
                # If we can't get the value, the parameter registry will handle it
                continue
        
        if not example_config:
            raise HTTPException(
                status_code=500,
                detail="No configuration values available from parameter registry"
            )
        
        return {
            "tenant_id": "tenant1",
            "verbose": api_config.verbose,
            "custom_name": api_config.default_custom_name,
            "resume_from_checkpoint": api_config.resume_from_checkpoint,
            "wandb_tags": api_config.default_wandb_tags.copy(),
            # Example payload should match field names used by the request
            "training_params": example_config,
            "model_params": {}
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate training example: {str(e)}"
        )

def get_parameter_documentation() -> Dict[str, Any]:
    """Get complete parameter documentation from registry"""
    try:
        docs = {}
        for param_name, param_def in parameter_registry.parameters.items():
            docs[param_name] = {
                "type": param_def.type.value,
                "description": param_def.description,
                "config_path": param_def.config_path,
                "cli_name": param_def.cli_name,
                "required": param_def.required
            }
            
            # Add current config value if available
            try:
                value = parameter_registry.get_config_value(param_name)
                if value is not None:
                    docs[param_name]["current_value"] = value
            except Exception:
                docs[param_name]["current_value"] = "Not available"
        
        return docs
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate parameter documentation: {str(e)}"
        )
