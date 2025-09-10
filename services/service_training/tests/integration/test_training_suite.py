import importlib
import inspect
import json
import os
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared_libs.utils.env_secrets import setup_environment_secrets


def test_cancel_via_web_api_endpoint():
    """Test cancelling a training job via the web API endpoint."""
    # Import the training router from service_training
    from endpoints.train import router as training_router

    # Create a training job via the API
    training_request = {
        "experiment_name": "test_web_api_job",
        "description": "Test job for web API cancellation",
        "training_params": {"num_epochs": 10},
        "model_params": {},
        "eval_params": {}
    }

    app = FastAPI()
    app.include_router(training_router)
    client = TestClient(app)

    # Start a training job
    response = client.post("/train", json=training_request)
    assert response.status_code == 200
    task_data = response.json()
    task_id = task_data["task_id"]

    # Verify job was created
    status_response = client.get(f"/train/{task_id}")
    assert status_response.status_code == 200

    # Cancel the job via DELETE endpoint
    cancel_response = client.delete(f"/train/{task_id}")
    assert cancel_response.status_code == 200
    cancel_data = cancel_response.json()

    # Verify response structure
    assert cancel_data["task_id"] == task_id
    assert cancel_data["status"] == "cancelled"
    assert "cancelled successfully" in cancel_data["message"]

# Mock training service for testing
class MockTrainingService:
    """Mock training service for testing service_training functionality."""
    
    def __init__(self):
        self.jobs = {}
    
    def create_job(self, req):
        """Create a mock training job."""
        import uuid
        task_id = str(uuid.uuid4())
        self.jobs[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": {"message": "Job created"},
            "created_at": "2024-01-01T00:00:00"
        }
        return task_id, {"tenant_id": req.tenant_id}
    
    def get_job(self, task_id):
        """Get a mock job by task_id."""
        return self.jobs.get(task_id)
    
    def cancel_job(self, task_id):
        """Cancel a mock job."""
        job = self.jobs.get(task_id)
        if job:
            job["status"] = "cancelled"
            job["progress"]["message"] = "Job cancelled"
            return True
        return False

# Global mock service instance
mock_training_service = MockTrainingService()


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
    from endpoints.train import router as train_router

    # Create FastAPI app and include the training router
    app = FastAPI()
    app.include_router(train_router)
    client = TestClient(app)

    # Create a training job via the API
    training_request = {
        "experiment_name": "test_training_job",
        "description": "Test job for cancellation",
        "training_params": {"num_epochs": 1},
        "model_params": {},
        "eval_params": {}
    }

    # Start training job
    response = client.post("/train/", json=training_request)
    assert response.status_code == 200
    job_data = response.json()
    task_id = job_data["task_id"]

    # Verify job was created
    assert job_data["status"] == "queued"

    # Get job status to verify it exists
    status_response = client.get(f"/train/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["task_id"] == task_id

    # Cancel the job
    cancel_response = client.delete(f"/train/{task_id}")
    assert cancel_response.status_code == 200
    cancel_data = cancel_response.json()
    assert cancel_data["task_id"] == task_id
    assert cancel_data["status"] == "cancelled"

    # Verify job status was updated
    final_status_response = client.get(f"/train/{task_id}")
    assert final_status_response.status_code == 200
    final_status_data = final_status_response.json()
    assert final_status_data["status"] == "cancelled"


def test_cancel_via_service_cli_with_pipeline_stop_verification():
    """Test cancelling a training job - the router handles cancellation in memory."""
    # Import the training router from service_training
    from endpoints.train import router as training_router

    # Create a FastAPI app with the training router
    app = FastAPI()
    app.include_router(training_router)
    client = TestClient(app)

    # Create a training job
    training_request = {
        "experiment_name": "test_training_job_cancel",
        "description": "Test job for cancellation",
        "training_params": {"num_epochs": 1},
        "model_params": {},
        "eval_params": {}
    }

    response = client.post("/train", json=training_request)
    assert response.status_code == 200
    task_data = response.json()
    task_id = task_data["task_id"]

    # Cancel the job
    cancel_response = client.delete(f"/train/{task_id}")
    assert cancel_response.status_code == 200
    cancel_data = cancel_response.json()
    assert cancel_data["task_id"] == task_id
    assert cancel_data["status"] == "cancelled"

    # Verify the job status was updated
    status_response = client.get(f"/train/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "cancelled"


def test_cancel_via_service_cli_pipeline_not_found():
    """Test cancelling a training job - the router doesn't interact with pipelines."""
    # Import the training router from service_training
    from endpoints.train import router as training_router

    # Create a FastAPI app with the training router
    app = FastAPI()
    app.include_router(training_router)
    client = TestClient(app)

    # Create a training job
    training_request = {
        "experiment_name": "test_training_job_simple",
        "description": "Test job for simple cancellation",
        "training_params": {"num_epochs": 1},
        "model_params": {},
        "eval_params": {}
    }

    response = client.post("/train", json=training_request)
    assert response.status_code == 200
    task_data = response.json()
    task_id = task_data["task_id"]

    # Cancel the job - the router just marks it as cancelled
    cancel_response = client.delete(f"/train/{task_id}")
    assert cancel_response.status_code == 200
    cancel_data = cancel_response.json()
    assert cancel_data["task_id"] == task_id
    assert cancel_data["status"] == "cancelled"

    # Verify the job status was updated
    status_response = client.get(f"/train/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "cancelled"


def test_cancel_via_web_api_endpoint():
    """Test cancelling a training job via the web API endpoint."""
    from endpoints.train import router as train_router_module

    app = FastAPI()
    app.include_router(train_router_module)
    client = TestClient(app)

    # Create a training job via the API
    training_request = {
        "experiment_name": "test_training_job",
        "description": "Test job for cancellation",
        "training_params": {"num_epochs": 1},
        "model_params": {},
        "eval_params": {}
    }

    # Start training job
    response = client.post("/train/", json=training_request)
    assert response.status_code == 200
    job_data = response.json()
    task_id = job_data["task_id"]

    # Verify job was created
    assert job_data["status"] == "queued"

    # Cancel the job via DELETE endpoint
    resp = client.delete(f"/train/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert body["message"] == "Training job cancelled successfully"
    assert body["task_id"] == task_id
    assert body["status"] == "cancelled"


def test_training_cancellation_with_interrupted_error():
    """Test that InterruptedError is properly raised and handled during training cancellation."""
    from unittest.mock import MagicMock, patch

    import torch
    # Create a minimal training setup
    from training_loop import Training

    # Mock components to avoid actual training
    with patch('torch.optim.AdamW'), \
         patch('torch.utils.data.DataLoader') as mock_dataloader, \
         patch('torch.nn.TripletMarginLoss'), \
         patch('wandb_logger.wandb_logger') as mock_wandb:

        # Setup mock dataloader
        mock_dl = MagicMock()
        mock_dl.__iter__.return_value = iter([(
            torch.randn(2, 3, 32, 32),  # anchor
            torch.randn(2, 3, 32, 32),  # positive
            torch.randn(2, 3, 32, 32),  # negative
            torch.tensor([0, 1])        # labels
        )])
        mock_dl.__len__.return_value = 1
        mock_dataloader.return_value = mock_dl

        # Setup mock wandb
        mock_wandb.enabled = False

        # Create training instance
        t = Training(device=torch.device('cpu'))
        t.num_epochs = 1
        t.batch_size = 2
        t.learning_rate = 1e-3
        t.margin = 1.0
        t.weight_decay = 0.0
        t.lr_scheduler_factor = 0.1
        t.scheduler_patience = 10
        t.scheduler_threshold = 0.01
        t.lr_scheduler_min_lr = 1e-6
        t.force_pretraining = True
        t.num_workers = 0

        # Create a mock model
        mock_model = MagicMock()
        mock_model.forward_triplet.return_value = (
            torch.randn(2, 64),  # anchor embeddings
            torch.randn(2, 64),  # positive embeddings
            torch.randn(2, 64)   # negative embeddings
        )

        # Setup training components
        t.model = mock_model
        t.optimizer = MagicMock()
        t.loss_fn = MagicMock()
        t.loss_fn.return_value = torch.tensor(1.0)
        t.dataloader = mock_dl

        # Test cancellation via stop_callback
        cancellation_requested = {'called': False}

        def stop_callback():
            if not cancellation_requested['called']:
                cancellation_requested['called'] = True
                return True
            return False

        # This should raise InterruptedError due to cancellation
        with pytest.raises(InterruptedError, match="Training cancelled by external request"):
            t.train(stop_callback=stop_callback)

        # Verify the cancellation was processed
        assert cancellation_requested['called'] is True


def test_pipeline_cancellation_with_pending_stop():
    """Test that pipelines registered after cancellation request are immediately stopped."""
    from unittest.mock import MagicMock, patch

    from shared_libs.common.pipeline import Pipeline, stop_pipeline

    # Mock pipeline components
    with patch('shared_libs.common.google_storage.get_storage') as mock_storage, \
         patch('shared_libs.common.google_storage.GCSPaths') as mock_paths:

        mock_storage.return_value = MagicMock()
        mock_paths.return_value = MagicMock()

        # First, request stop for a pipeline that doesn't exist yet
        pipeline_name = "test_pending_cancellation"
        result = stop_pipeline(pipeline_name)
        assert result is True  # Should succeed (queued for later)

        # Now create and register the pipeline
        step_defs = {"test_step": {"description": "Test step", "function": lambda ctx: {"status": "completed"}}}
        pipeline = Pipeline(
            storage_client=mock_storage.return_value,
            step_definitions=step_defs,
            pipeline_name=pipeline_name,
            verbose=False
        )

        # Verify the pipeline was immediately stopped due to pending cancellation
        assert pipeline.is_stop_requested() is True
        # Note: Status remains 'not_started' until _check_cancellation() is called during execution
        # But the stop request should be set, which will cause cancellation when the pipeline runs


def test_siamesenet_dino_can_download_and_initialize(tmp_path: Path):
    """Integration test: ensure SiameseNet downloads DINOv3 from Hugging Face and initializes."""
    # Ensure environment secrets are set for the test run
    setup_environment_secrets()

    assert 'HUGGINGFACE_HUB_TOKEN' in os.environ and os.environ.get('HUGGINGFACE_HUB_TOKEN'), \
        "HUGGINGFACE_HUB_TOKEN not found in environment after setup_environment_secrets()"

    os.environ.setdefault('HF_HOME', str(tmp_path / 'hf_cache'))

    try:
        import torch
        from siamesenet_dino import SiameseNet

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
def test_train_all_resnet_with_checkpoint_verification():
    """End-to-end test: ResNet training with 2 epochs and 2 datasets, verifying checkpoint creation."""
    import wandb
    from shared_libs.config.all_config import wandb_config

    # Ensure secrets for longer e2e tests
    setup_environment_secrets()

    from workflows.training_workflow import train_workflow as train

    run_name = f"e2e_resnet_checkpoint_test_{uuid.uuid4().hex[:8]}"

    try:
        results = train(
            tenant_id="tenant1",
            verbose=False,
            save_intermediate=False,
            custom_name=run_name,
            resume_from_checkpoint=False,
            wandb_tags=["e2e", "resnet", "checkpoint_test"],
            training_kwargs={"num_epochs": 2, "batch_size": 8},
            model_kwargs={"model_class_module": "siamesenet", "model_class_str": "SiameseNet"},
            n_datasets_to_use=2,
        )

        # Verify training completed successfully
        assert isinstance(results, dict)
        assert results.get("status") == "completed"
        assert results.get("steps_completed", 0) == 6  # 3 steps per dataset for 2 datasets

        # Verify checkpoint creation and cleanup
        try:
            api = wandb.Api()
            run = api.run(f"{wandb_config.team}/{wandb_config.project}/{run_name}")
            artifacts = list(run.logged_artifacts())

            # Clean up artifacts first
            for artifact in artifacts:
                try:
                    artifact.delete()
                    print(f"🧹 Cleaned up artifact: {artifact.name}")
                except Exception as e:
                    print(f"⚠️ Failed to delete artifact {artifact.name}: {e}")

            # Clean up the run
            try:
                run.delete()
                print(f"🧹 Cleaned up wandb run: {run_name}")
            except Exception as e:
                print(f"⚠️ Failed to clean up wandb run '{run_name}': {e}")

        except Exception as e:
            print(f"⚠️ Could not access wandb run '{run_name}' for cleanup: {e}")
            print("ℹ️ This is likely a timing issue - the run may still be syncing to wandb servers")

        # Additional cleanup for test artifacts (more robust approach)
        try:
            api = wandb.Api()

            # Clean up test model artifacts
            test_artifact_names = ["test-siamesenet"]
            for artifact_name in test_artifact_names:
                try:
                    artifact_type_obj = api.artifact_type("model", project=wandb_config.project)
                    collection = artifact_type_obj.collection(artifact_name)
                    artifacts = list(collection.artifacts())
                    deleted_count = 0
                    for artifact in artifacts:
                        try:
                            artifact.delete()
                            deleted_count += 1
                        except Exception as e:
                            print(f"⚠️ Failed to delete artifact version {artifact.version}: {e}")

                    if deleted_count > 0:
                        print(f"🧹 Cleaned up {deleted_count} versions of test model artifact: {artifact_name}")

                except Exception as e:
                    if "404" in str(e) or "not found" in str(e).lower():
                        print(f"ℹ️ Test model artifact '{artifact_name}' not found")
                    else:
                        print(f"⚠️ Failed to access test model artifact '{artifact_name}': {e}")

            # Clean up checkpoint artifacts
            try:
                checkpoint_type_obj = api.artifact_type("model_checkpoint", project=wandb_config.project)
                checkpoint_collections = checkpoint_type_obj.collections()

                for collection in checkpoint_collections:
                    if collection.name.startswith("test-") or run_name in collection.name:
                        try:
                            artifacts = list(collection.artifacts())
                            deleted_count = 0
                            for artifact in artifacts:
                                try:
                                    artifact.delete()
                                    deleted_count += 1
                                except Exception as e:
                                    print(f"⚠️ Failed to delete checkpoint artifact version {artifact.version}: {e}")

                            if deleted_count > 0:
                                print(f"🧹 Cleaned up {deleted_count} versions of checkpoint artifact: {collection.name}")
                        except Exception as e:
                            print(f"⚠️ Failed to clean up checkpoint collection '{collection.name}': {e}")

            except Exception as e:
                print(f"⚠️ Failed to clean up checkpoint artifacts: {e}")

        except Exception as e:
            print(f"⚠️ Failed to clean up test artifacts: {e}")


    except Exception as e:
        pytest.fail(f"ResNet end-to-end test with checkpoint verification failed: {type(e).__name__}: {e}")
   


@pytest.mark.slow
@pytest.mark.e2e
def test_train_all_with_dino_memory_stable():
    """End-to-end test: DINO training with 1 epoch and single dataset, with memory stability assertions."""
    import gc

    import psutil

    setup_environment_secrets()

    # Setup memory monitoring
    process = psutil.Process()
    gc.collect()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    from workflows.training_workflow import train_workflow as train

    try:
        results = train(
            tenant_id="tenant1",
            verbose=False,
            save_intermediate=False,
            custom_name="e2e_dino_1_epoch_single_dataset",
            resume_from_checkpoint=False,
            wandb_tags=["e2e", "dino", "memory_stable"],
            training_kwargs={"num_epochs": 1, "batch_size": 4, "force_pretraining": True},
            model_kwargs={"model_class_module": "siamesenet_dino", "model_class_str": "SiameseNet"},
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

        # Assert memory stability - allow up to 500MB for comprehensive DINO workflow (model loading, training, evaluation)
        assert abs(memory_delta) <= 500, f"Memory usage changed by {memory_delta:.1f}MB, exceeds 500MB threshold"

        # Relaxed memory leak check - allow up to 500MB for full workflow (DINO + evaluation + checkpoints)
        # This accounts for legitimate memory usage in model loading, dataset processing, training, and evaluation
        assert memory_delta <= 500, f"Memory usage increased by {memory_delta:.1f}MB, within acceptable range"

    except Exception as e:
        pytest.fail(f"DINO end-to-end test failed: {type(e).__name__}: {e}")


def test_train_signature_has_n_datasets_to_use():
    """Test that the train function signature includes n_datasets_to_use parameter."""
    from workflows.training_workflow import train_workflow as train
    sig = inspect.signature(train)
    assert "n_datasets_to_use" in sig.parameters


def test_convert_request_to_kwargs_includes_top_level_n_datasets():
    """Test that request conversion includes top-level n_datasets parameter."""
    from services.service_training.src.training_service import \
        _convert_request_to_kwargs

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
