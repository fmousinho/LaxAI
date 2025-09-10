
import os
import time
import uuid
from pathlib import Path

import pytest
import torch

import wandb
from services.service_training.src.wandb_logger import WandbLogger
from shared_libs.config.all_config import wandb_config
from shared_libs.utils.env_secrets import setup_environment_secrets

# Ensure secrets are loaded for wandb communication
setup_environment_secrets()


@pytest.fixture
def wandb_logger_instance() -> WandbLogger:
    """Fixture to provide a configured WandbLogger instance."""
    logger = WandbLogger(enabled=True)
    return logger


@pytest.mark.slow
def test_checkpoint_subprocess_upload_e2e(wandb_logger_instance: WandbLogger, tmp_path: Path):
    """
    A focused integration test to verify the checkpoint upload subprocess.

    It tests the following sequence:
    1. Initialize a wandb run.
    2. Create a dummy checkpoint file.
    3. Trigger the async checkpoint upload method.
    4. Wait for the upload to complete.
    5. Verify the artifact exists in the wandb registry.
    6. Clean up the created run and artifacts.
    """
    logger = wandb_logger_instance
    run_name = f"e2e_checkpoint_subprocess_test_{uuid.uuid4().hex[:8]}"
    run_id = None
    api = wandb.Api()

    try:
        # 1. Initialize a wandb run
        config = {"test_type": "checkpoint_subprocess"}
        logger.init_run(config=config, run_name=run_name, tags=["e2e", "checkpoint_test"])
        assert logger.run is not None
        run_id = logger.run.id
        print(f"🚀 Started wandb run: {run_name} (ID: {run_id})")

        # 2. Create a dummy checkpoint file
        checkpoint_path = tmp_path / "dummy_checkpoint.pth"
        torch.save({"epoch": 1, "model_state_dict": {}}, checkpoint_path)
        print(f"📄 Created dummy checkpoint file at: {checkpoint_path}")

        # 3. Trigger the async checkpoint upload
        # This is the core method we want to test. We call the private method
        # directly to test its logic in isolation.
        checkpoint_name = logger.get_checkpoint_name()
        logger._trigger_checkpoint_upload_and_cleanup(
            checkpoint_path=str(checkpoint_path),
            checkpoint_name=checkpoint_name,
            epoch=1,
            loss=0.5
        )
        print("⏳ Triggered checkpoint upload subprocess. Waiting for completion...")

        # 4. Wait for the upload to complete
        # In a real scenario, training would continue. Here, we wait for futures.
        logger._wait_for_pending_operations()  # Wait for completion
        print("✅ Async upload process finished.")

        # 5. Verify the artifact exists in the wandb registry
        checkpoint_name = logger.get_checkpoint_name()
        artifact_path = f"{wandb_config.team}/{wandb_config.project}/{checkpoint_name}:latest"
        
        print(f"🔍 Verifying artifact exists at: {artifact_path}")
        try:
            artifact = api.artifact(artifact_path, type="model_checkpoint")
            assert artifact is not None
            # The artifact name includes the version, so check if it starts with our checkpoint name
            assert artifact.name.startswith(checkpoint_name), f"Expected artifact name to start with '{checkpoint_name}', got '{artifact.name}'"
            print(f"✅ Verification successful! Found artifact '{artifact.name}'")
        except wandb.errors.CommError as e:
            pytest.fail(f"Failed to find artifact '{artifact_path}'. It was not created. Error: {e}")

    except Exception as e:
        pytest.fail(f"Test failed during execution: {type(e).__name__}: {e}")

    finally:
        # 6. Clean up the created run and artifacts
        if logger.run:
            logger.finish()
            print(f"🏁 Finished wandb run: {run_name}")

        if run_id:
            try:
                run_path = f"{wandb_config.team}/{wandb_config.project}/{run_id}"
                run = api.run(run_path)
                run.delete()
                print(f"🧹 Cleaned up wandb run: {run.name}")
            except Exception as e:
                print(f"⚠️ Failed to clean up wandb run '{run_id}': {e}")

        # Clean up test artifacts
        try:
            # Delete the test-siamesenet model artifact
            artifact_path = f"{wandb_config.team}/{wandb_config.project}/test-siamesenet:latest"
            artifact = api.artifact(artifact_path, type="model")
            if artifact:
                artifact.delete()
                print(f"🧹 Cleaned up test model artifact: test-siamesenet")
        except Exception as e:
            print(f"⚠️ Failed to clean up test model artifact 'test-siamesenet': {e}")
