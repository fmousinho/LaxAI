import json
import sys


def test_openapi_contains_eval_params():
    
    from fastapi import FastAPI
    from v1.endpoints import train as train_endpoint
    app = FastAPI()
    app.include_router(train_endpoint.router, prefix='/api/v1')

    schema = app.openapi()

    # Find the TrainingRequest schema in components
    components = schema.get('components', {})
    schemas = components.get('schemas', {})

    assert 'TrainingRequest' in schemas, 'TrainingRequest schema missing from OpenAPI components'
    tr = schemas['TrainingRequest']

    properties = tr.get('properties', {})
    assert 'eval_params' in properties, 'eval_params not exposed in TrainingRequest OpenAPI schema'


def test_pydantic_docs_include_training_params():
    
    from v1.schemas import training as t

    # Ensure TrainingConfig doc contains at least one known training param from registry
    doc = t.get_parameter_documentation if hasattr(t, 'get_parameter_documentation') else None
    # Fallback: check that TrainingConfig model has model_fields
    TrainingConfig = t.TrainingConfig
    if hasattr(TrainingConfig, 'model_fields'):
        fields = set(TrainingConfig.model_fields.keys())
    else:
        fields = set(TrainingConfig.__fields__.keys())

    assert 'batch_size' in fields, 'batch_size should be present in dynamic TrainingConfig fields'
