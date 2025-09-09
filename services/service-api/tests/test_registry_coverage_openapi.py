import sys


def test_all_registered_params_present_in_openapi():
    """Ensure every parameter in parameter_registry is exposed in the appropriate OpenAPI schema.

    Rules:
      - If param is in parameter_registry.training_parameters -> TrainingConfig
      - If param is in parameter_registry.model_parameters -> ModelConfig
      - If param is in parameter_registry.eval_parameters -> EvalConfig
    """
    

    from config.parameter_registry import parameter_registry
    from fastapi import FastAPI
    from v1.endpoints import train as train_endpoint

    # Build a lightweight app with only the train router to generate OpenAPI safely
    app = FastAPI()
    app.include_router(train_endpoint.router, prefix='/api/v1')
    schema = app.openapi()

    components = schema.get('components', {})
    schemas = components.get('schemas', {})

    # Helper to get properties set for a schema
    def schema_props(name):
        s = schemas.get(name, {})
        return set(s.get('properties', {}).keys())

    training_props = schema_props('TrainingConfig')
    model_props = schema_props('ModelConfig')
    eval_props = schema_props('EvalConfig')

    missing = []

    for name, param in parameter_registry.parameters.items():
        if name in parameter_registry.training_parameters:
            if name not in training_props:
                missing.append((name, 'TrainingConfig'))
        elif name in parameter_registry.model_parameters:
            if name not in model_props:
                missing.append((name, 'ModelConfig'))
        elif name in parameter_registry.eval_parameters:
            if name not in eval_props:
                missing.append((name, 'EvalConfig'))
        else:
            # If not categorized, fall back to overall parameters mapping (treat as training)
            if name not in training_props:
                missing.append((name, 'TrainingConfig (fallback)'))

    if missing:
        missing_lines = '\n'.join([f"{n} -> expected in {s}" for n, s in missing])
        raise AssertionError(f"Missing parameter mappings in OpenAPI schemas:\n{missing_lines}")
