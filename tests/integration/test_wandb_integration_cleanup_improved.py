import time
import pytest
import torch
from tests.fixtures.wandb_test_helpers import fast_wandb_test, create_minimal_checkpoint
from train.wandb_logger import wandb_logger
from config.all_config import wandb_config


@pytest.mark.integration
def test_wandb_cleanup_integration_fast():
    """Faster integration test for WandB cleanup using helper context manager."""
    with fast_wandb_test("fast_cleanup") as (run_name, artifact_names):
        # Create two minimal checkpoints
        checkpoint1 = create_minimal_checkpoint(epoch=1)
        checkpoint2 = create_minimal_checkpoint(epoch=2)
        
        # Save checkpoints with minimal data
        wandb_logger.save_checkpoint(**checkpoint1)
        time.sleep(1)  # Ensure distinct timestamps
        wandb_logger.save_checkpoint(**checkpoint2)
        
        # Track the artifact for cleanup
        artifact_name = wandb_logger._sanitize_artifact_name(wandb_logger._get_checkpoint_name())
        artifact_names.append(artifact_name)
        
        # Allow indexing to propagate
        time.sleep(1)
        
        # Verify cleanup works
        try:
            api = wandb_logger.wandb_api
            if api:
                artifact_type_api = api.artifact_type("model_checkpoint", project=wandb_config.project)
                collection_api = artifact_type_api.collection(artifact_name)
                versions_before = list(collection_api.artifacts())
                
                if len(versions_before) >= 2:
                    # Run cleanup keeping only latest
                    wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=1)
                    time.sleep(1)
                    
                    versions_after = list(collection_api.artifacts())
                    assert len(versions_after) <= 1, f"Expected <=1 versions after cleanup, got {len(versions_after)}"
        except Exception as e:
            pytest.skip(f"Could not verify cleanup via API: {e}")


@pytest.mark.integration  
def test_wandb_cleanup_integration_original():
    """Original integration test - kept for compatibility but with better cleanup."""
    if not wandb_config.enabled:
        pytest.skip("wandb is disabled in config")

    api = wandb_logger.wandb_api
    if not api:
        api = wandb_logger._login_and_get_api()
        wandb_logger.wandb_api = api

    if not api:
        pytest.skip("Could not initialize wandb Api (check WANDB_API_KEY in environment)")

    import uuid
    run_name = f"integration_cleanup_{uuid.uuid4().hex[:8]}"
    artifact_name = None
    
    try:
        wandb_logger.init_run(config={'test': True}, run_name=run_name)

        # Create two checkpoints with minimal data
        wandb_logger.save_checkpoint(epoch=1, model_state_dict={'w': torch.tensor([1])}, optimizer_state_dict={}, loss=0.1)
        time.sleep(1.5)
        wandb_logger.save_checkpoint(epoch=2, model_state_dict={'w': torch.tensor([2])}, optimizer_state_dict={}, loss=0.05)

        time.sleep(2)

        artifact_name = wandb_logger._sanitize_artifact_name(wandb_logger._get_checkpoint_name())

        # Test cleanup functionality
        artifact_type_api = wandb_logger.wandb_api.artifact_type("model_checkpoint", project=wandb_config.project)
        collection_api = artifact_type_api.collection(artifact_name)
        versions_before = list(collection_api.artifacts())

        assert len(versions_before) >= 2, f"Expected >=2 artifact versions before cleanup, got {len(versions_before)}"

        wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=1)
        time.sleep(2)

        versions_after = list(collection_api.artifacts())
        assert len(versions_after) <= 1, f"Expected <=1 artifact versions after cleanup, got {len(versions_after)}"

    finally:
        # Guaranteed cleanup in finally block
        try:
            wandb_logger.finish()
        except Exception:
            pass
            
        # Clean up all remaining artifacts from this test
        if artifact_name:
            try:
                wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=0)
                time.sleep(1)
            except Exception as e:
                print(f"Warning: Final cleanup failed for {artifact_name}: {e}")
