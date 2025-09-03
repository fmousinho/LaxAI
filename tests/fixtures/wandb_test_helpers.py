import pytest
import uuid
import time
from contextlib import contextmanager
from typing import List, Optional
from train.wandb_logger import wandb_logger
from config.all_config import wandb_config


class WandbTestManager:
    """Manages WandB artifacts and cleanup for tests."""
    
    def __init__(self):
        self.created_artifacts = []
        self.created_runs = []
        
    def track_artifact(self, artifact_name: str):
        """Track an artifact for cleanup."""
        self.created_artifacts.append(artifact_name)
        
    def track_run(self, run_id: str):
        """Track a run for cleanup."""
        self.created_runs.append(run_id)
        
    def cleanup_all(self):
        """Clean up all tracked artifacts and runs."""
        # Clean up artifacts
        for artifact_name in self.created_artifacts:
            try:
                wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=0)
                time.sleep(0.5)  # Brief pause for propagation
            except Exception as e:
                print(f"Warning: Failed to cleanup artifact {artifact_name}: {e}")
                
        # Clean up runs if needed (implement if required)
        for run_id in self.created_runs:
            try:
                # WandB doesn't easily support run deletion via API
                # This is mainly for tracking purposes
                pass
            except Exception as e:
                print(f"Warning: Failed to cleanup run {run_id}: {e}")
                
        self.created_artifacts.clear()
        self.created_runs.clear()


@pytest.fixture
def wandb_test_manager():
    """Pytest fixture that provides WandB test management with automatic cleanup."""
    manager = WandbTestManager()
    yield manager
    # Cleanup happens automatically after test completes
    manager.cleanup_all()


@contextmanager
def fast_wandb_test(run_name_prefix: str = "test"):
    """Context manager for fast WandB tests with automatic cleanup."""
    if not wandb_config.enabled:
        pytest.skip("WandB is disabled in config")
        
    run_name = f"{run_name_prefix}_{uuid.uuid4().hex[:8]}"
    artifact_names = []
    
    try:
        # Initialize run
        wandb_logger.init_run(config={'test': True}, run_name=run_name)
        yield run_name, artifact_names
        
    finally:
        # Always cleanup, even if test fails
        try:
            wandb_logger.finish()
        except Exception:
            pass
            
        # Clean up any artifacts created during test
        for artifact_name in artifact_names:
            try:
                wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=0)
                time.sleep(0.2)  # Minimal pause for propagation
            except Exception as e:
                print(f"Warning: Failed to cleanup artifact {artifact_name}: {e}")


def create_minimal_checkpoint(epoch: int = 1):
    """Create a minimal checkpoint for testing purposes - follows test config defaults."""
    from tests.unit.test_config import get_minimal_checkpoint_data
    return get_minimal_checkpoint_data(epoch)


def mock_wandb_checkpoint_save(monkeypatch, checkpoint_data: dict):
    """Mock checkpoint saving to avoid actual WandB uploads in fast tests."""
    def mock_save_checkpoint(*args, **kwargs):
        # Return a mock artifact reference without actually uploading
        class MockArtifact:
            def __init__(self):
                self.name = f"mock_checkpoint_{uuid.uuid4().hex[:8]}"
                self.version = "v0"
        return MockArtifact()
    
    monkeypatch.setattr(wandb_logger, 'save_checkpoint', mock_save_checkpoint)
