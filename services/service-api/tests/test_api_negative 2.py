import os
import importlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


try:
    import utils.env_secrets as env_secrets
    env_secrets.setup_environment_secrets()
except Exception:
    # let tests handle missing secrets gracefully
    pass


def make_app_with_router():
    from v1.endpoints import train as train_router_module
    app = FastAPI()
    app.include_router(train_router_module.router)
    return app


def test_post_train_invalid_types_returns_422():
    """Send invalid types in the training request and expect 422 validation error."""
    app = make_app_with_router()
    client = TestClient(app)

    payload = {
        # tenant_id should be a string; intentionally send integer
        "tenant_id": 12345,
        # n_datasets_to_use should be int or null; send a string to break validation
        "n_datasets_to_use": "not_an_int",
    }

    resp = client.post('/train', json=payload)
    # FastAPI/Pydantic should return 422 Unprocessable Entity for validation errors
    assert resp.status_code == 422, f"Expected 422 but got {resp.status_code}: {resp.text}"


def test_post_train_malformed_json_returns_400_and_no_crash():
    """Send malformed JSON and assert the API responds with 400 (Bad Request) and not 500."""
    app = make_app_with_router()
    client = TestClient(app)

    # Intentionally malformed JSON (missing quotes/braces)
    bad_body = '{tenant_id: not-quoted}'

    resp = client.post('/train', data=bad_body, headers={'Content-Type': 'application/json'})
    assert resp.status_code in (400, 422), f"Expected client error (400/422), got {resp.status_code}"


def test_get_status_for_unknown_task_returns_404():
    app = make_app_with_router()
    client = TestClient(app)

    resp = client.get('/train/nonexistent_task/status')
    assert resp.status_code == 404


def test_delete_nonexistent_task_returns_404():
    app = make_app_with_router()
    client = TestClient(app)

    resp = client.delete('/train/nonexistent_task')
    assert resp.status_code == 404
