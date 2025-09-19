"""
Centralized parameter registry for training, model, and API configuration.
This module provides a single source of truth for all parameters used across
CLI arguments, API schemas, and training code.
"""

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field


class ParameterType(Enum):
    """Parameter types for validation and conversion"""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STR = "str"
    LIST_STR = "list[str]"


class ParameterCategory(Enum):
    """Parameter categories for filtering"""

    TRAINING = "training"
    MODEL = "model"
    SYSTEM = "system"


@dataclass
class ParameterDefinition:
    """Complete definition of a training parameter"""

    name: str
    type: ParameterType
    description: str
    config_path: str  # e.g., "training_config.learning_rate"
    category: ParameterCategory = ParameterCategory.TRAINING  # Parameter category for filtering
    cli_name: Optional[str] = None  # Custom CLI arg name, defaults to --{name.replace('_', '-')}
    api_field_kwargs: Dict[str, Any] = field(default_factory=dict)  # Extra Pydantic Field kwargs
    cli_kwargs: Dict[str, Any] = field(default_factory=dict)  # Extra argparse kwargs
    required: bool = False  # Whether required when not in config

    def __post_init__(self):
        if self.cli_name is None:
            self.cli_name = f"--{self.name.replace('_', '-')}"


class ParameterRegistry:
    """Central registry for all training parameters"""

    def __init__(self):
        self.parameters: Dict[str, ParameterDefinition] = {}
        self.training_parameters: Dict[str, ParameterDefinition] = {}
        self.model_parameters: Dict[str, ParameterDefinition] = {}
        # Container for evaluation-specific parameters (populated by register_eval_param)
        self.eval_parameters: Dict[str, ParameterDefinition] = {}
        self._register_default_parameters()

    def register(self, param: ParameterDefinition):
        """Register a parameter definition"""
        self.parameters[param.name] = param

    def register_training_param(self, param: ParameterDefinition):
        """Register a training parameter"""
        self.parameters[param.name] = param
        self.training_parameters[param.name] = param

    def register_model_param(self, param: ParameterDefinition):
        """Register a model parameter"""
        self.parameters[param.name] = param
        self.model_parameters[param.name] = param

    def register_eval_param(self, param: ParameterDefinition):
        """Register an evaluation parameter"""
        self.parameters[param.name] = param
        self.eval_parameters[param.name] = param

    def get_training_parameters(self) -> Dict[str, ParameterDefinition]:
        """Get only training parameters"""
        return self.training_parameters.copy()

    def get_model_parameters(self) -> Dict[str, ParameterDefinition]:
        """Get only model parameters"""
        return self.model_parameters.copy()

    def generate_cli_parser_for_training(
        self, parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        """Generate argparse arguments for training parameters only"""
        return self._generate_cli_parser_for_params(self.training_parameters.values(), parser)

    def generate_cli_parser_for_model(
        self, parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        """Generate argparse arguments for model parameters only"""
        return self._generate_cli_parser_for_params(self.model_parameters.values(), parser)

    def generate_cli_parser(
        self, parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        """Generate argparse arguments for all parameters (training, model, eval)"""
        # Start with training parameters
        parser = self._generate_cli_parser_for_params(self.training_parameters.values(), parser)
        # Add model parameters
        parser = self._generate_cli_parser_for_params(self.model_parameters.values(), parser)
        # Add eval parameters
        parser = self._generate_cli_parser_for_params(self.eval_parameters.values(), parser)
        return parser

    def generate_pydantic_fields_for_training(self) -> Dict[str, Any]:
        """Generate Pydantic field definitions for training parameters only"""
        return self._generate_pydantic_fields_for_params(self.training_parameters.values())

    def generate_pydantic_fields_for_model(self) -> Dict[str, Any]:
        """Generate Pydantic field definitions for model parameters only"""
        return self._generate_pydantic_fields_for_params(self.model_parameters.values())

    def generate_pydantic_fields_for_eval(self) -> Dict[str, Any]:
        """Generate Pydantic field definitions for evaluation parameters only"""
        return self._generate_pydantic_fields_for_params(self.eval_parameters.values())

    def get_config_value(self, param_name: str) -> Any:
        """Get the default value from config for a parameter"""
        if param_name not in self.parameters:
            raise ValueError(f"Unknown parameter: {param_name}")

        param = self.parameters[param_name]
        config_path_parts = param.config_path.split(".")

        # Dynamic import based on config path
        if config_path_parts[0] == "training_config":
            from config.all_config import training_config as config_obj
        elif config_path_parts[0] == "model_config":
            from config.all_config import model_config as config_obj
        elif config_path_parts[0] == "evaluator_config":
            from config.all_config import evaluator_config as config_obj
        elif config_path_parts[0] == "wandb_config":
            from config.all_config import wandb_config as config_obj
        else:
            raise ValueError(f"Unknown config object: {config_path_parts[0]}")

        # Navigate to the config value
        value = config_obj
        for part in config_path_parts[1:]:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                if param.required:
                    raise ValueError(
                        f"Required parameter {param_name} not found in config at {param.config_path}"
                    )
                return None

        return value

    def get_kwarg_or_config(self, param_name: str, kwargs: Dict[str, Any]) -> Any:
        """
        Get parameter value from kwargs or config, with proper error handling.
        This replaces the get_kwarg_or_config function in Training class.
        """
        # First check kwargs
        if param_name in kwargs:
            return kwargs[param_name]

        # Then check if parameter is registered and get from config
        if param_name in self.parameters:
            try:
                value = self.get_config_value(param_name)
                if value is not None:
                    return value
            except Exception:
                pass

        # If we get here, parameter is missing
        param_def = self.parameters.get(param_name)
        if param_def and param_def.required:
            raise ValueError(
                f"Missing required hyperparameter '{param_name}' in kwargs and config."
            )
        else:
            raise ValueError(
                f"Missing required hyperparameter '{param_name}' in kwargs and config."
            )

    def generate_cli_parser(
        self, parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        """Generate argparse arguments from parameter definitions"""
        if parser is None:
            parser = argparse.ArgumentParser()

        for param in self.parameters.values():
            kwargs = {"help": param.description, "dest": param.name, **param.cli_kwargs}

            # Set type conversion
            if param.type == ParameterType.FLOAT:
                kwargs["type"] = float
            elif param.type == ParameterType.INT:
                kwargs["type"] = int
            elif param.type == ParameterType.BOOL:
                kwargs["action"] = (
                    "store_true" if not kwargs.get("default", False) else "store_false"
                )
            elif param.type == ParameterType.LIST_STR:
                kwargs["nargs"] = "+"
                kwargs["type"] = str

            # Add default from config if available
            try:
                default_value = self.get_config_value(param.name)
                # Respect an explicit default provided via param.cli_kwargs
                if "default" not in kwargs and default_value is not None:
                    kwargs["default"] = default_value
                    kwargs["help"] += f" (default: {default_value})"
            except Exception:
                if param.required:
                    kwargs["required"] = True

            parser.add_argument(param.cli_name or f"--{param.name}", **kwargs)

        return parser

    def generate_pydantic_fields(self) -> Dict[str, Any]:
        """Generate Pydantic field definitions from parameter registry"""
        fields = {}

        for param in self.parameters.values():
            # Get default value from config
            try:
                default_value = self.get_config_value(param.name)
            except Exception:
                default_value = ... if param.required else None

            # Convert ParameterType to Python type
            if param.type == ParameterType.FLOAT:
                python_type = float
            elif param.type == ParameterType.INT:
                python_type = int
            elif param.type == ParameterType.BOOL:
                python_type = bool
            elif param.type == ParameterType.STR:
                python_type = str
            elif param.type == ParameterType.LIST_STR:
                python_type = List[str]
            else:
                python_type = Any

            # Make optional if not required
            if not param.required and default_value is None:
                python_type = Optional[python_type]

            # Create Field with description and default
            field_kwargs = {"description": param.description, **param.api_field_kwargs}

            if default_value is not ...:
                field_kwargs["default"] = default_value

            fields[param.name] = (python_type, Field(**field_kwargs))

        return fields

    def get_parameter_documentation(self) -> str:
        """Generate documentation string for all parameters"""
        doc_lines = ["Training Parameters:"]

        for param in self.parameters.values():
            type_str = param.type.value
            try:
                default_val = self.get_config_value(param.name)
                default_str = f" (default: {default_val})" if default_val is not None else ""
            except Exception:
                default_str = " (required)" if param.required else ""

            doc_lines.append(f"    {param.name} ({type_str}): {param.description}{default_str}")

        return "\n".join(doc_lines)

    def _generate_cli_parser_for_params(
        self, params, parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        """Generate argparse arguments for a specific set of parameters"""
        import argparse

        if parser is None:
            parser = argparse.ArgumentParser()

        for param in params:
            kwargs = {"help": param.description, "dest": param.name, **param.cli_kwargs}

            # Set type conversion
            if param.type == ParameterType.FLOAT:
                kwargs["type"] = float
            elif param.type == ParameterType.INT:
                kwargs["type"] = int
            elif param.type == ParameterType.BOOL:
                kwargs["action"] = (
                    "store_true" if not kwargs.get("default", False) else "store_false"
                )
            elif param.type == ParameterType.LIST_STR:
                kwargs["nargs"] = "+"
                kwargs["type"] = str

            # Add argument
            cli_name = param.cli_name or f"--{param.name.replace('_', '-')}"
            parser.add_argument(cli_name, **kwargs)

        return parser

    def _generate_pydantic_fields_for_params(self, params) -> Dict[str, Any]:
        """Generate Pydantic field definitions for a specific set of parameters"""
        fields = {}

        for param in params:
            # Get default value from config
            try:
                default_value = self.get_config_value(param.name)
            except Exception:
                default_value = ... if param.required else None

            # Convert ParameterType to Python type
            if param.type == ParameterType.FLOAT:
                python_type = float
            elif param.type == ParameterType.INT:
                python_type = int
            elif param.type == ParameterType.BOOL:
                python_type = bool
            elif param.type == ParameterType.STR:
                python_type = str
            elif param.type == ParameterType.LIST_STR:
                python_type = List[str]
            else:
                python_type = Any

            # Make optional if not required
            if not param.required and default_value is None:
                python_type = Optional[python_type]

            # Create Field with description and default
            field_kwargs = {"description": param.description, **param.api_field_kwargs}

            if default_value is not ...:
                field_kwargs["default"] = default_value

            fields[param.name] = (python_type, Field(**field_kwargs))

        return fields

    def _register_default_parameters(self):
        """Register all the training parameters"""

        # Training parameters
        training_params = [
            ParameterDefinition(
                name="num_epochs",
                type=ParameterType.INT,
                description="Number of training epochs",
                config_path="training_config.num_epochs",
            ),
            ParameterDefinition(
                name="batch_size",
                type=ParameterType.INT,
                description="Batch size for training",
                config_path="training_config.batch_size",
            ),
            ParameterDefinition(
                name="num_workers",
                type=ParameterType.INT,
                description="Number of DataLoader workers for parallel data loading",
                config_path="training_config.num_workers",
            ),
            ParameterDefinition(
                name="learning_rate",
                type=ParameterType.FLOAT,
                description="Learning rate for optimizer",
                config_path="training_config.learning_rate",
            ),
            ParameterDefinition(
                name="margin",
                type=ParameterType.FLOAT,
                description="Margin for triplet loss",
                config_path="training_config.margin",
            ),
            ParameterDefinition(
                name="weight_decay",
                type=ParameterType.FLOAT,
                description="Weight decay for optimizer",
                config_path="training_config.weight_decay",
            ),
            ParameterDefinition(
                name="lr_scheduler_patience",
                type=ParameterType.INT,
                description="Patience for learning rate scheduler",
                config_path="training_config.lr_scheduler_patience",
            ),
            ParameterDefinition(
                name="lr_scheduler_threshold",
                type=ParameterType.FLOAT,
                description="Threshold for learning rate scheduler",
                config_path="training_config.lr_scheduler_threshold",
            ),
            ParameterDefinition(
                name="lr_scheduler_min_lr",
                type=ParameterType.FLOAT,
                description="Minimum learning rate for scheduler",
                config_path="training_config.lr_scheduler_min_lr",
            ),
            ParameterDefinition(
                name="lr_scheduler_factor",
                type=ParameterType.FLOAT,
                description="Factor for LR scheduler",
                config_path="training_config.lr_scheduler_factor",
            ),
            ParameterDefinition(
                name="force_pretraining",
                type=ParameterType.BOOL,
                description="Force pretrained ResNet defaults",
                config_path="training_config.force_pretraining",
            ),
            ParameterDefinition(
                name="early_stopping_patience",
                type=ParameterType.INT,
                description="Early stopping patience",
                config_path="training_config.early_stopping_patience",
            ),
            ParameterDefinition(
                name="min_images_per_player",
                type=ParameterType.INT,
                description="Minimum images required for a player to participate in the training",
                config_path="training_config.min_images_per_player",
            ),
            ParameterDefinition(
                name="prefetch_factor",
                type=ParameterType.INT,
                description="Prefetch factor for data loading",
                config_path="training_config.prefetch_factor",
            ),
            ParameterDefinition(
                name="margin_decay_rate",
                type=ParameterType.FLOAT,
                description="Rate at which to decay the triplet loss margin",
                config_path="training_config.margin_decay_rate",
            ),
            ParameterDefinition(
                name="margin_change_threshold",
                type=ParameterType.FLOAT,
                description="Minimum change in margin to trigger an update",
                config_path="training_config.margin_change_threshold",
            ),
            ParameterDefinition(
                name="train_ratio",
                type=ParameterType.FLOAT,
                description="Percentage of dataset samples to be used for training (versus validation)",
                config_path="training_config.train_ratio",
            ),
            ParameterDefinition(
                name="n_datasets_to_use",
                type=ParameterType.INT,
                description="Number of datasets to use for training (optional)",
                config_path="training_config.n_datasets_to_use",
                required=False,
            ),
            ParameterDefinition(
                name="wandb_project",
                type=ParameterType.STR,
                description="WandB project name",
                config_path="wandb_config.project",
            ),
        ]

        # Model parameters
        model_params = [
            ParameterDefinition(
                name="embedding_dim",
                type=ParameterType.INT,
                description="Embedding dimension",
                config_path="model_config.embedding_dim",
            ),
            ParameterDefinition(
                name="dropout_rate",
                type=ParameterType.FLOAT,
                description="Dropout rate",
                config_path="model_config.dropout_rate",
            ),
            ParameterDefinition(
                name="input_height",
                type=ParameterType.INT,
                description="Required height for input images",
                config_path="model_config.input_height",
            ),
            ParameterDefinition(
                name="input_width",
                type=ParameterType.INT,
                description="Required width for input images",
                config_path="model_config.input_width",
            ),
            ParameterDefinition(
                name="enable_grass_mask",
                type=ParameterType.BOOL,
                description="Determines whether grass will be removed when creating the dataset",
                config_path="model_config.enable_grass_mask",
            ),
            ParameterDefinition(
                name="model_class_module",
                type=ParameterType.STR,
                description="The module where the model class is defined",
                config_path="model_config.model_class_module",
            ),
            ParameterDefinition(
                name="model_class",
                type=ParameterType.STR,
                description="The name of the model class",
                config_path="model_config.model_class_str",
            ),
        ]

        eval_params = [
            ParameterDefinition(
                name="number_of_workers",
                type=ParameterType.INT,
                description="Number of workers for DataLoader",
                config_path="evaluator_config.number_of_workers",
            ),
            ParameterDefinition(
                name="emb_batch_size",
                type=ParameterType.INT,
                description="Batch size for embedding generation",
                config_path="evaluator_config.emb_batch_size",
            ),
            ParameterDefinition(
                name="eval_prefetch_factor",
                type=ParameterType.INT,
                description="Number of batches to prefetch for DataLoader",
                config_path="evaluator_config.prefetch_factor",
            ),
        ]

        # Register training parameters
        for param in training_params:
            self.register_training_param(param)

        # Register model parameters
        for param in model_params:
            self.register_model_param(param)

        for param in eval_params:
            self.register_eval_param(param)


# Global instance
parameter_registry = ParameterRegistry()

# Sanity-check: ensure registry implements expected registration methods.
# This will raise a clear AttributeError early if the object was replaced or
# incorrectly instantiated in a deployed image (helps catch stale/faulty builds).
required_methods = [
    "register",
    "register_training_param",
    "register_model_param",
    "register_eval_param",
    "generate_pydantic_fields_for_eval",
]
missing = [m for m in required_methods if not hasattr(parameter_registry, m)]
if missing:
    raise RuntimeError(
        f"ParameterRegistry missing required methods: {missing}. Did a previous fallback or partial initialization run in this environment?"
    )
