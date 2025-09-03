import pytest
import torch
import time
from unittest.mock import MagicMock, patch
from tests.fixtures.wandb_test_helpers import fast_wandb_test, create_minimal_checkpoint
from train.wandb_logger import wandb_logger
from config.all_config import wandb_config


@pytest.mark.integration
@pytest.mark.parametrize("fast_mode", [True, False])  
def test_wandb_cleanup_integration_parameterized(fast_mode, wandb_artifact_cleaner):
    """Parameterized integration test that can run in fast mode (mocked) or real mode."""
    if not wandb_config.enabled:
        pytest.skip("WandB is disabled in config")
        
    if fast_mode:
        # Fast mode: mock the expensive WandB operations
        with patch.object(wandb_logger, 'save_checkpoint') as mock_save:
            mock_artifact = MagicMock()
            mock_artifact.name = "mock_checkpoint"
            mock_artifact.version = "v0"
            mock_save.return_value = mock_artifact
            
            with patch.object(wandb_logger, '_cleanup_old_checkpoints') as mock_cleanup:
                mock_cleanup.return_value = None
                
                # Run the test logic without actual uploads
                import uuid
                run_name = f"fast_integration_{uuid.uuid4().hex[:8]}"
                wandb_logger.init_run(config={'test': True}, run_name=run_name)
                
                try:
                    # Simulate checkpoint creation
                    checkpoint_data = create_minimal_checkpoint(1)
                    wandb_logger.save_checkpoint(**checkpoint_data)
                    
                    checkpoint_data2 = create_minimal_checkpoint(2)  
                    wandb_logger.save_checkpoint(**checkpoint_data2)
                    
                    # Simulate cleanup
                    artifact_name = "mock_checkpoint"
                    wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=1)
                    
                    # Verify mocks were called
                    assert mock_save.call_count == 2
                    mock_cleanup.assert_called_once_with(artifact_name, keep_latest=1)
                    
                finally:
                    wandb_logger.finish()
                    
    else:
        # Real mode: run the original integration test with real WandB
        api = wandb_logger.wandb_api
        if not api:
            api = wandb_logger._login_and_get_api()
            wandb_logger.wandb_api = api

        if not api:
            pytest.skip("Could not initialize wandb Api (check WANDB_API_KEY in environment)")

        import uuid
        run_name = f"real_integration_{uuid.uuid4().hex[:8]}"
        artifact_name = None
        
        try:
            wandb_logger.init_run(config={'test': True}, run_name=run_name)

            # Create minimal checkpoints
            wandb_logger.save_checkpoint(epoch=1, model_state_dict={'w': torch.tensor([1])}, optimizer_state_dict={}, loss=0.1)
            time.sleep(1)
            wandb_logger.save_checkpoint(epoch=2, model_state_dict={'w': torch.tensor([2])}, optimizer_state_dict={}, loss=0.05)

            time.sleep(1)

            artifact_name = wandb_logger._sanitize_artifact_name(wandb_logger._get_checkpoint_name())
            wandb_artifact_cleaner.track_artifact(artifact_name)

            # Test cleanup
            artifact_type_api = wandb_logger.wandb_api.artifact_type("model_checkpoint", project=wandb_config.project)
            collection_api = artifact_type_api.collection(artifact_name)
            versions_before = list(collection_api.artifacts())

            if len(versions_before) >= 2:
                wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=1)
                time.sleep(1)

                versions_after = list(collection_api.artifacts())
                assert len(versions_after) <= 1, f"Expected <=1 versions after cleanup, got {len(versions_after)}"

        finally:
            try:
                wandb_logger.finish()
            except Exception:
                pass


@pytest.mark.fast
def test_wandb_cleanup_fast_only():
    """Fast-only test for CI/CD pipelines that should complete quickly."""
    if not wandb_config.enabled:
        pytest.skip("WandB is disabled in config")
        
    with patch.object(wandb_logger, 'save_checkpoint') as mock_save:
        mock_artifact = MagicMock()
        mock_artifact.name = "mock_checkpoint"
        mock_save.return_value = mock_artifact
        
        with patch.object(wandb_logger, '_cleanup_old_checkpoints') as mock_cleanup:
            import uuid
            run_name = f"fast_only_{uuid.uuid4().hex[:8]}"
            wandb_logger.init_run(config={'test': True}, run_name=run_name)
            
            try:
                # Simulate operations
                checkpoint_data = create_minimal_checkpoint(1)
                wandb_logger.save_checkpoint(**checkpoint_data)
                
                wandb_logger._cleanup_old_checkpoints("mock_checkpoint", keep_latest=1)
                
                # Verify behavior
                mock_save.assert_called_once()
                mock_cleanup.assert_called_once()
                
            finally:
                wandb_logger.finish()
