import sys


def test_training_config_includes_registered_params():
    """Ensure dynamic TrainingConfig model includes all parameters registered under 'training_config'."""
    

    from config.parameter_registry import parameter_registry
    from v1.schemas.training import create_training_config_model

    TrainingConfig = create_training_config_model()

    # pydantic v2 uses model_fields, v1 uses __fields__
    if hasattr(TrainingConfig, 'model_fields'):
        model_field_names = set(TrainingConfig.model_fields.keys())
    else:
        model_field_names = set(TrainingConfig.__fields__.keys())

    # Collect parameters that belong to training_config
    registered_training_params = set(
        name for name, p in parameter_registry.parameters.items()
        if getattr(p, 'config_path', '').startswith('training_config')
    )

    missing = registered_training_params - model_field_names

    assert not missing, f"These training params are registered but missing from TrainingConfig: {sorted(missing)}"
