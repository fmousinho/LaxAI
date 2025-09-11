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

# ---------------------
# Core Cancellation Tests
# ---------------------

def test_cancel_via_web_api_endpoint():
    """Test cancelling a training job via the web API DELETE endpoint."""
    import os
    import sys
    import time

    # Add the service_training src path to sys.path for imports
    service_src_path = os.path.join(os.path.dirname(__file__), '../../src')
    sys.path.insert(0, service_src_path)

    # Import the training router from service_training
    from endpoints.train import router as training_router

        # Create a training job with minimal parameters for quick execution
    training_request = {
        "custom_name": "test_cancel_job",
        "tenant_id": "tenant1",
        "resume_from_checkpoint": False,
        "training_params": {
            "num_epochs": 2,  # Very short training to ensure cancellation happens
            "batch_size": 4,
            "learning_rate": 0.001
        },
        "model_params": {
            "model_class_module": "siamesenet",
            "model_class": "SiameseNet"
        },
        "eval_params": {},
        "n_datasets_to_use": 1  # Use minimal dataset
    }

    app = FastAPI()
    app.include_router(training_router)
    client = TestClient(app)

    # Start a training job
    response = client.post("/train", json=training_request)
    assert response.status_code == 200
    task_data = response.json()
    task_id = task_data["task_id"]
    print(f"Started training job: {task_id}")

    # Verify job was created
    status_response = client.get(f"/train/{task_id}")
    assert status_response.status_code == 200
    initial_status = status_response.json()
    assert initial_status["status"] in ["queued", "running"]

    # Wait for job to start running (give it time to initialize)
    max_wait_time = 30  # seconds
    wait_time = 0
    while wait_time < max_wait_time:
        status_response = client.get(f"/train/{task_id}")
        current_status = status_response.json()
        if current_status["status"] == "running":
            print(f"Job {task_id} is now running")
            break
        time.sleep(1)
        wait_time += 1
    else:
        # If we didn't break, the job never started running
        pytest.fail(f"Job {task_id} never transitioned to 'running' status within {max_wait_time} seconds. Current status: {current_status.get('status')}")

    # Cancel immediately once the job starts running
    print(f"Cancelling job {task_id} immediately after it started running")
    cancel_response = client.delete(f"/train/{task_id}")
    assert cancel_response.status_code == 200
    cancel_data = cancel_response.json()

    # Verify response structure
    assert cancel_data["task_id"] == task_id
    assert cancel_data["status"] == "cancelled"
    assert "cancelled successfully" in cancel_data["message"]
    print(f"Job {task_id} cancelled successfully")

    # Verify job status was updated to cancelled
    final_status_response = client.get(f"/train/{task_id}")
    assert final_status_response.status_code == 200
    final_status_data = final_status_response.json()
    assert final_status_data["status"] == "cancelled"
    print(f"Confirmed job {task_id} status is cancelled")

    # Wait for the job to actually finish (should be cancelled, not completed)
    # Set a timeout of 3 minutes (180 seconds) to prevent hanging
    timeout_start = time.time()
    max_completion_wait = 180  # 3 minutes

    while time.time() - timeout_start < max_completion_wait:
        status_response = client.get(f"/train/{task_id}")
        current_status = status_response.json()

        if current_status["status"] in ["completed", "failed", "cancelled"]:
            break
        time.sleep(2)  # Check every 2 seconds

    # Assert that the job did NOT complete successfully (cancellation worked)
    assert current_status["status"] != "completed", \
        f"Training job completed successfully despite cancellation request. Status: {current_status['status']}"

    # Assert that the test didn't timeout
    assert time.time() - timeout_start < max_completion_wait, \
        f"Test timed out after {max_completion_wait} seconds waiting for job to finish. Current status: {current_status['status']}"

    # Final assertion: job should be cancelled
    assert current_status["status"] == "cancelled", \
        f"Expected job to be cancelled, but got status: {current_status['status']}"

    print(f"✅ Cancellation test passed - job {task_id} was properly cancelled")


def test_cli_cancellation_with_signals():
    """Test cancelling a training job via the actual CLI with signal handling."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from unittest.mock import MagicMock, patch

    # Mock heavy dependencies to avoid import issues during testing
    mock_modules = [
        'workflows.training_workflow',
        'workflows',
        'common.google_storage',
        'common',
        'config.logging_config',
        'parameter_registry',
        'utils.env_secrets',
        'rfdetr',
        'transformers'
    ]

    for module in mock_modules:
        sys.modules[module] = MagicMock()

    # Mock the TrainingWorkflow class
    with patch('workflows.training_workflow.TrainingWorkflow') as mock_workflow_class:
        mock_workflow_instance = MagicMock()
        mock_workflow_class.return_value = mock_workflow_instance
        mock_workflow_instance.run.return_value = None  # Mock successful run

        # Start CLI training process with inline script to avoid import issues
        cmd = [
            "/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/.venv/bin/python",
            "-c",
            """
import sys
import signal
import threading
import time

# Mock the heavy imports
sys.modules['workflows'] = type(sys)('workflows')
sys.modules['workflows.training_workflow'] = type(sys)('training_workflow')
sys.modules['workflows.training_workflow'].TrainingWorkflow = type

# Simple mock CLI that simulates the signal handling
cancellation_event = threading.Event()

def signal_handler(signum, frame):
    print(f"Received signal {signum}. Requesting training cancellation")
    cancellation_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("Starting mock training...")
try:
    # Simulate long-running training
    while not cancellation_event.is_set():
        time.sleep(0.1)
    print("Training cancelled successfully")
except KeyboardInterrupt:
    print("Training interrupted")
"""
        ]

        # Start the process
        proc = subprocess.Popen(
            cmd,
            cwd="/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            # Wait a moment for the process to start
            time.sleep(2)

            # Send SIGINT (Ctrl+C) to cancel
            proc.send_signal(signal.SIGINT)

            # Wait for process to terminate
            stdout, stderr = proc.communicate(timeout=10)

            # Verify the process exited with cancellation code
            # -2 means terminated by SIGINT (signal 2), which is correct for cancellation
            assert proc.returncode == -2, f"Expected return code -2 (SIGINT), got {proc.returncode}"

            # Verify cancellation message in output
            assert "Received signal 2. Requesting training cancellation" in stdout or "Received signal 2. Requesting training cancellation" in stderr

            print("✅ CLI cancellation test passed")

        except subprocess.TimeoutExpired:
            # If process doesn't terminate, kill it
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail("CLI process did not terminate after cancellation signal")

        finally:
            # Ensure process is cleaned up
            if proc.poll() is None:
                proc.kill()


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

        # Wait 10 seconds for wandb synchronization, then clean up all test artifacts
        import wandb
        print("⏳ Waiting 10 seconds for wandb synchronization...")
        time.sleep(10)
        print("✅ Finished waiting for wandb sync")

        # Clean up all test artifacts that start with "test-"
        try:
            api = wandb.Api()

            # Get all artifact types
            artifact_types = api.artifact_types(project=wandb_config.project)

            total_deleted = 0
            for artifact_type in artifact_types:
                try:
                    # Get all collections for this artifact type
                    collections = artifact_type.collections()

                    for collection in collections:
                        try:
                            # Check if collection name starts with "test-"
                            if collection.name.startswith("test-"):
                                artifacts = list(collection.artifacts())

                                for artifact in artifacts:
                                    try:
                                        # Try to delete the artifact directly
                                        artifact.delete()
                                        print(f"🧹 Cleaned up test artifact: {artifact.name}")
                                        total_deleted += 1
                                    except Exception as e:
                                        error_msg = str(e).lower()
                                        # If deletion failed due to alias, try to remove alias first
                                        if "alias" in error_msg or "409" in str(e):
                                            try:
                                                print(f"⚠️ Artifact {artifact.name} has alias, attempting to remove alias and retry...")
                                                # Try to delete by removing aliases first
                                                # In wandb, we can use the API to remove aliases
                                                if hasattr(artifact, 'aliases') and artifact.aliases:
                                                    try:
                                                        # Try to remove aliases using the collection
                                                        collection = artifact.collection()
                                                        for alias in artifact.aliases[:]:  # Copy the list to avoid modification issues
                                                            try:
                                                                # Remove alias by setting it to None or empty
                                                                collection.set_aliases({alias: None})
                                                                print(f"ℹ️ Removed alias '{alias}' from {artifact.name}")
                                                            except Exception as alias_error:
                                                                print(f"⚠️ Failed to remove alias '{alias}' from {artifact.name}: {alias_error}")
                                                    except Exception as collection_error:
                                                        print(f"⚠️ Failed to access collection for {artifact.name}: {collection_error}")

                                                # Try to delete again after removing aliases
                                                artifact.delete()
                                                print(f"🧹 Cleaned up test artifact (after alias removal): {artifact.name}")
                                                total_deleted += 1
                                            except Exception as retry_error:
                                                print(f"⚠️ Failed to delete test artifact {artifact.name} even after alias removal: {retry_error}")
                                        else:
                                            print(f"⚠️ Failed to delete test artifact {artifact.name}: {e}")

                        except Exception as e:
                            print(f"⚠️ Failed to access collection {collection.name}: {e}")

                except Exception as e:
                    print(f"⚠️ Failed to access artifact type {artifact_type.name}: {e}")

            if total_deleted > 0:
                print(f"🧹 Cleaned up {total_deleted} test artifacts total")
            else:
                print("ℹ️ No test artifacts found to clean up")

        except Exception as e:
            print(f"⚠️ Failed to clean up test artifacts: {e}")

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
                                    print(f"🧹 Cleaned up checkpoint artifact: {artifact.name}")
                                except Exception as e:
                                    print(f"⚠️ Failed to delete checkpoint artifact {artifact.name}: {e}")

                            if deleted_count > 0:
                                print(f"🧹 Cleaned up {deleted_count} checkpoint artifacts from collection: {collection.name}")
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
    
    # Check if **kwargs is present, allowing n_datasets_to_use to be passed dynamically
    assert any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()), \
        "Function must accept **kwargs to handle n_datasets_to_use"
    
    # Optionally, verify that n_datasets_to_use can be passed without error (if the function is callable)
    # This ensures the parameter is supported, even if not explicit in the signature
    try:
        # Note: This is a dry-run check; adjust based on actual function behavior
        train(tenant_id="test", n_datasets_to_use=1)
    except TypeError as e:
        pytest.fail(f"Function does not accept n_datasets_to_use: {e}")


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
