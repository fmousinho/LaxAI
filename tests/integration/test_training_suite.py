import os
import json
import time
import uuid
import inspect
import subprocess
import tempfile
import signal
from typing import Optional
from pathlib import Path
import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.utils.env_secrets import setup_environment_secrets

# Ensure environment secrets for integration tests that require them
try:
    setup_environment_secrets()
except Exception:
    # Let tests that require secrets handle failures explicitly
    pass


# ---------------------
# Helpers and fixtures
# ---------------------

class DummyRequest:
    """Minimal request object with attribute access and model_dump()."""
    def __init__(self, tenant_id: str = "tenant1", n_datasets_to_use: Optional[int] = None):
        """Initialize DummyRequest with default test values."""
        self.tenant_id = tenant_id
        self.verbose = True
        self.custom_name = "run1"
        self.resume_from_checkpoint = True
        self.wandb_tags = ["tag1"]
        self.n_datasets_to_use = n_datasets_to_use
        self.training_params = None
        self.model_params = None

    def model_dump(self):
        """Return dictionary representation of the request object."""
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


def make_request_obj(tenant_id: str = "tenant1", n_datasets_to_use: Optional[int] = None):
    """Create a DummyRequest object with the specified parameters."""
    return DummyRequest(tenant_id=tenant_id, n_datasets_to_use=n_datasets_to_use)


# ---------------------
# Tests merged
# ---------------------

def test_cancel_via_service_cli():
    """Test cancelling a training job via the service CLI interface."""
    # Import the training service module the router uses so the in-memory
    # job store is shared.
    training_service = importlib.import_module('services.training_service')
    # Also import the router module for API endpoint tests
    from src.api.v1.endpoints import train as train_router_module

    # Create a job via the service create_job helper
    req = make_request_obj()
    task_id, kwargs = training_service.create_job(req)

    # Ensure job exists and is pending
    job = training_service.get_job(task_id)
    assert job is not None
    assert job["status"] == "pending"

    # Call cancel_job directly (CLI-style / service API) and assert it updates state
    result = training_service.cancel_job(task_id)

    job_after = training_service.get_job(task_id)
    assert job_after is not None
    assert job_after["status"] in ("cancelled", "cancelling")


def test_cancel_via_web_api_endpoint():
    """Test cancelling a training job via the web API endpoint."""
    training_service = importlib.import_module('services.training_service')
    from src.api.v1.endpoints import train as train_router_module

    req = make_request_obj()
    task_id, kwargs = training_service.create_job(req)

    app = FastAPI()
    app.include_router(train_router_module.router)
    client = TestClient(app)

    resp = client.delete(f"/train/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert body["message"].startswith("Training job")

    job_after = training_service.get_job(task_id)
    assert job_after is not None
    assert job_after["status"] in ("cancelled", "cancelling")


def test_siamesenet_dino_can_download_and_initialize(tmp_path: Path):
    """Integration test: ensure SiameseNet downloads DINOv3 from Hugging Face and initializes."""
    # Ensure environment secrets are set for the test run
    setup_environment_secrets()

    assert 'HUGGINGFACE_HUB_TOKEN' in os.environ and os.environ.get('HUGGINGFACE_HUB_TOKEN'), \
        "HUGGINGFACE_HUB_TOKEN not found in environment after setup_environment_secrets()"

    os.environ.setdefault('HF_HOME', str(tmp_path / 'hf_cache'))

    try:
        from src.train.siamesenet_dino import SiameseNet
        import torch

        net = SiameseNet()
        net.eval()
        with torch.no_grad():
            h, w = getattr(net.backbone, '_expected_input_size', (224, 224))
            x = torch.randn(1, 3, h, w)
            emb = net(x)

        assert emb is not None
    except Exception as e:
        pytest.fail(f"SiameseNet failed to download/initialize: {type(e).__name__}: {e}")


@pytest.mark.slow
@pytest.mark.e2e
def test_train_all_resnet_with_two_datasets_memory_stable():
    """End-to-end test: ResNet training with 2 epochs and 2 datasets, with memory stability assertions."""
    import psutil
    import gc

    # Ensure secrets for longer e2e tests
    setup_environment_secrets()

    # Setup memory monitoring
    process = psutil.Process()
    gc.collect()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    from scripts.train_all import train

    try:
        results = train(
            tenant_id="tenant1",
            verbose=False,
            save_intermediate=False,
            custom_name="e2e_resnet_2_epochs_2_datasets",
            resume_from_checkpoint=False,
            wandb_tags=["e2e", "resnet", "memory_stable"],
            training_kwargs={"num_epochs": 2, "batch_size": 8},
            model_kwargs={"model_class_module": "train.siamesenet", "model_class_str": "SiameseNet"},
            n_datasets_to_use=2,  # Two datasets as requested
        )

        # Verify training completed successfully
        assert isinstance(results, dict)
        assert results.get("status") == "completed"
        assert results.get("steps_completed", 0) == 3  # All pipeline steps

        # Memory stability assertions
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory

        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")

        # Assert memory stability - should not increase by more than 200MB
        assert abs(memory_delta) <= 200, f"Memory usage changed by {memory_delta:.1f}MB, exceeds 200MB threshold"

        # Assert no memory leaks (should not increase by more than 50MB)
        assert memory_delta <= 50, f"Potential memory leak detected: +{memory_delta:.1f}MB"

    except Exception as e:
        pytest.fail(f"ResNet end-to-end test failed: {type(e).__name__}: {e}")


@pytest.mark.slow
@pytest.mark.e2e
def test_train_all_with_dino_memory_stable():
    """End-to-end test: DINO training with 1 epoch and single dataset, with memory stability assertions."""
    import psutil
    import gc

    setup_environment_secrets()

    # Setup memory monitoring
    process = psutil.Process()
    gc.collect()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    from scripts.train_all import train

    try:
        results = train(
            tenant_id="tenant1",
            verbose=False,
            save_intermediate=False,
            custom_name="e2e_dino_1_epoch_single_dataset",
            resume_from_checkpoint=False,
            wandb_tags=["e2e", "dino", "memory_stable"],
            training_kwargs={"num_epochs": 1, "batch_size": 4, "force_pretraining": True},
            model_kwargs={"model_class_module": "train.siamesenet_dino", "model_class_str": "SiameseNet"},
            n_datasets_to_use=1,
        )

        # Verify training completed successfully
        assert isinstance(results, dict)
        assert results.get("status") == "completed"
        assert results.get("steps_completed", 0) == 3  # All pipeline steps

        # Memory stability assertions
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory

        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")

        # Assert memory stability - should not increase by more than 150MB (DINO downloads models)
        assert abs(memory_delta) <= 150, f"Memory usage changed by {memory_delta:.1f}MB, exceeds 150MB threshold"

        # Assert no memory leaks (should not increase by more than 30MB for single epoch)
        assert memory_delta <= 30, f"Potential memory leak detected: +{memory_delta:.1f}MB"

    except Exception as e:
        pytest.fail(f"DINO end-to-end test failed: {type(e).__name__}: {e}")


def test_train_signature_has_n_datasets_to_use():
    """Test that the train function signature includes n_datasets_to_use parameter."""
    from src.scripts.train_all import train
    sig = inspect.signature(train)
    assert "n_datasets_to_use" in sig.parameters


def test_convert_request_to_kwargs_includes_top_level_n_datasets():
    """Test that request conversion includes top-level n_datasets parameter."""
    from src.services.training_service import _convert_request_to_kwargs

    req = SimpleNamespace(
        tenant_id="tenant1",
        verbose=True,
        custom_name="run1",
        resume_from_checkpoint=True,
        wandb_tags=["tag1"],
        n_datasets_to_use=5,
        training_params=None,
        model_params=None,
    )

    kwargs = _convert_request_to_kwargs(req)

    assert "n_datasets_to_use" in kwargs
    assert kwargs["n_datasets_to_use"] == 5
    assert "training_kwargs" not in kwargs
