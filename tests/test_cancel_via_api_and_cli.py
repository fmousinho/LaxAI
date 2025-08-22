from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import importlib

# Import the same module path the API router uses so the in-memory job store is
# shared. The router imports `services.training_service` (top-level package), so
# import that module here.
training_service = importlib.import_module('services.training_service')
from src.api.v1.endpoints import train as train_router_module


class DummyRequest:
    """Minimal request object with attribute access and model_dump()."""
    def __init__(self, tenant_id="tenant1", n_datasets_to_use=None):
        self.tenant_id = tenant_id
        self.verbose = True
        self.custom_name = "run1"
        self.resume_from_checkpoint = True
        self.wandb_tags = ["tag1"]
        self.n_datasets_to_use = n_datasets_to_use
        self.training_params = None
        self.model_params = None

    def model_dump(self):
        # Return a dict representation similar to Pydantic's .model_dump()
        return {
            "tenant_id": self.tenant_id,
            "verbose": self.verbose,
            "custom_name": self.custom_name,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "wandb_tags": self.wandb_tags,
            "n_datasets_to_use": self.n_datasets_to_use,
            "training_params": self.training_params,
            "model_params": self.model_params,
        }


def make_request_obj(tenant_id="tenant1", n_datasets_to_use=None):
    return DummyRequest(tenant_id=tenant_id, n_datasets_to_use=n_datasets_to_use)


def test_cancel_via_service_cli():
    # Create a job via the service create_job helper
    req = make_request_obj()
    task_id, kwargs = training_service.create_job(req)

    # Ensure job exists and is pending
    job = training_service.get_job(task_id)
    assert job is not None
    assert job["status"] == "pending"

    # Call cancel_job directly (CLI-style / service API) and assert it updates state
    result = training_service.cancel_job(task_id)

    # result may be True if stop_pipeline returned True, or False; in both cases
    # the job status should be set to cancelled/cancelling.
    job_after = training_service.get_job(task_id)
    assert job_after is not None
    assert job_after["status"] in ("cancelled", "cancelling")


def test_cancel_via_web_api_endpoint():
    # Create a separate job
    req = make_request_obj()
    task_id, kwargs = training_service.create_job(req)

    # Mount the router into a small FastAPI app for testing
    app = FastAPI()
    app.include_router(train_router_module.router)
    client = TestClient(app)

    # Call DELETE /train/{task_id}
    resp = client.delete(f"/train/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert body["message"].startswith("Training job")

    # The job status should be cancelled/cancelling after the API call
    job_after = training_service.get_job(task_id)
    assert job_after is not None
    assert job_after["status"] in ("cancelled", "cancelling")
