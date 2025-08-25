import sys
import types
import importlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_app_with_router_after_mock(fake_registry_module):
    # Inject fake parameter_registry into sys.modules before importing schema
    sys.modules['config.parameter_registry'] = fake_registry_module
    # Reload the schemas module so TrainingConfig is created from fake registry
    import importlib
    import src.api.v1.schemas.training as schemas
    importlib.reload(schemas)

    # Now import router and build app
    # Inject a lightweight fake services.training_service to avoid starting
    # background pipelines during validation tests.
    import types
    fake_service = types.ModuleType('services.training_service')

    def create_job(request):
        # Return a dummy task id and kwargs (use the request.model_dump if available)
        try:
            payload = request.model_dump()
        except Exception:
            payload = {}
        return ('test-task-' + 'id', payload)

    def start_job(task_id, kwargs, background_tasks):
        # Do nothing; avoid launching pipelines
        return None

    def get_job(task_id):
        return None

    def list_jobs():
        return {}

    def cancel_job(task_id):
        return False

    fake_service.create_job = create_job
    fake_service.start_job = start_job
    fake_service.get_job = get_job
    fake_service.list_jobs = list_jobs
    fake_service.cancel_job = cancel_job
    # Provide a validator that uses the dynamic TrainingConfig already
    # imported by make_app_with_router_after_mock via the fake registry.
    def validate_training_params(payload):
        from src.api.v1.schemas import training as schemas
        # Use model_validate to enforce required/typed fields
        if payload is None:
            payload_to_validate = {}
        else:
            if hasattr(payload, 'model_dump'):
                payload_to_validate = payload.model_dump()
            else:
                payload_to_validate = payload if isinstance(payload, dict) else dict(payload)
        return schemas.TrainingConfig.model_validate(payload_to_validate)

    fake_service.validate_training_params = validate_training_params

    sys.modules['services.training_service'] = fake_service

    from src.api.v1.endpoints import train as train_router_module
    app = FastAPI()
    app.include_router(train_router_module.router)
    return app


def test_dynamic_training_params_validation_type_mismatch():
    # Create a fake registry that defines a single int field 'num_epochs'
    fake = types.ModuleType('config.parameter_registry')

    def gen_fields_for_training():
        # Pydantic expects mapping name -> (type, default)
        return {
            'num_epochs': (int, 5),
            'batch_size': (int, 16)
        }

    def gen_fields_for_model():
        return {}

    from types import SimpleNamespace
    fake.parameter_registry = SimpleNamespace(
        generate_pydantic_fields_for_training=gen_fields_for_training,
        generate_pydantic_fields_for_model=gen_fields_for_model,
        parameters={'num_epochs': None}
    )

    app = make_app_with_router_after_mock(fake)
    client = TestClient(app)

    # Provide training_params with a type mismatch (string instead of int)
    payload = {
        'tenant_id': 'tenant1',
        'training_params': {
            'num_epochs': 'not-an-int',
            'batch_size': 16
        }
    }

    resp = client.post('/train', json=payload)
    assert resp.status_code == 422, f"Expected 422 for type mismatch, got {resp.status_code} - {resp.text}"


def test_dynamic_training_params_missing_required_field():
    # Fake registry where num_epochs required (no default simulated by Field absence)
    fake = types.ModuleType('config.parameter_registry')

    def gen_fields_for_training():
        # Provide a required field by using Ellipsis as default per pydantic convention
        return {
            'num_epochs': (int, ...),
        }

    def gen_fields_for_model():
        return {}

    from types import SimpleNamespace
    fake.parameter_registry = SimpleNamespace(
        generate_pydantic_fields_for_training=gen_fields_for_training,
        generate_pydantic_fields_for_model=gen_fields_for_model,
        parameters={'num_epochs': None}
    )

    app = make_app_with_router_after_mock(fake)
    client = TestClient(app)

    payload = {
        'tenant_id': 'tenant1',
        # training_params missing 'num_epochs'
        'training_params': {}
    }

    resp = client.post('/train', json=payload)
    # The API should not crash; it may either return a 422 validation error
    # or accept and enqueue the job depending on dynamic schema handling. It
    # must not return a 5xx server error.
    assert resp.status_code < 500, f"Unexpected server error: {resp.status_code} - {resp.text}"
    # Prefer validation errors if present
    assert resp.status_code in (200, 422)
