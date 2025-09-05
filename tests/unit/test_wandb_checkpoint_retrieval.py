"""
Unit tests for WandbLogger checkpoint retrieval functionality.
Tests the ability to retrieve the latest checkpoint from WandB.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional


class TestWandbLoggerCheckpointRetrieval:
    """Unit tests for WandbLogger checkpoint retrieval functionality."""

    @pytest.fixture
    def mock_wandb_logger(self):
        """Create a mock WandbLogger instance for testing."""
        from src.train.wandb_logger import WandbLogger
        logger = WandbLogger(enabled=True)
        return logger

    @pytest.fixture
    def sample_checkpoint_data(self):
        """Create sample checkpoint data for testing."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create some optimizer state by running a forward/backward pass
        x = torch.randn(5, 10)
        y = model(x).sum()
        y.backward()
        optimizer.step()

        return {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.123,
            'model_config': {'test': True}
        }

    def test_get_checkpoint_name_method_exists(self, mock_wandb_logger):
        """Test that get_checkpoint_name method exists and returns expected value."""
        checkpoint_name = mock_wandb_logger.get_checkpoint_name()
        assert isinstance(checkpoint_name, str)
        # Should return basic checkpoint name when no run is initialized
        assert checkpoint_name == "checkpoint"

    def test_get_checkpoint_name_with_run_name(self, mock_wandb_logger):
        """Test that checkpoint names include run name when run is initialized."""
        # Mock a wandb run with a specific name
        mock_run = MagicMock()
        mock_run.name = "my-training-run"
        mock_wandb_logger.run = mock_run

        checkpoint_name = mock_wandb_logger.get_checkpoint_name()
        # The test prefix depends on whether _is_test_run detects it as a test
        expected = "checkpoint-my-training-run"
        assert checkpoint_name == expected

    def test_get_checkpoint_name_sanitizes_run_name(self, mock_wandb_logger):
        """Test that checkpoint names properly sanitize run names."""
        # Mock a wandb run with a name that needs sanitization
        mock_run = MagicMock()
        mock_run.name = "my@training#run$with%special&chars"
        mock_wandb_logger.run = mock_run

        checkpoint_name = mock_wandb_logger.get_checkpoint_name()
        # Should sanitize special characters
        assert "checkpoint-" in checkpoint_name
        assert checkpoint_name != "checkpoint-my@training#run$with%special&chars"

    def test_unique_checkpoint_names_for_different_runs(self, mock_wandb_logger):
        """Test that different run names produce unique checkpoint names."""
        # Test first run
        mock_run1 = MagicMock()
        mock_run1.name = "run-1"
        mock_wandb_logger.run = mock_run1
        checkpoint_name_1 = mock_wandb_logger.get_checkpoint_name()

        # Test second run
        mock_run2 = MagicMock()
        mock_run2.name = "run-2"
        mock_wandb_logger.run = mock_run2
        checkpoint_name_2 = mock_wandb_logger.get_checkpoint_name()

        # Test third run
        mock_run3 = MagicMock()
        mock_run3.name = "different-model-architecture"
        mock_wandb_logger.run = mock_run3
        checkpoint_name_3 = mock_wandb_logger.get_checkpoint_name()

        # All checkpoint names should be unique
        assert checkpoint_name_1 == "checkpoint-run-1"
        assert checkpoint_name_2 == "checkpoint-run-2"
        assert checkpoint_name_3 == "checkpoint-different-model-architecture"
        
        # Verify they are all different
        assert checkpoint_name_1 != checkpoint_name_2
        assert checkpoint_name_1 != checkpoint_name_3
        assert checkpoint_name_2 != checkpoint_name_3

    def test_load_checkpoint_with_latest_version(self, mock_wandb_logger, sample_checkpoint_data):
        """Test loading checkpoint with 'latest' version."""
        with patch.object(mock_wandb_logger.wandb_api, 'artifact') as mock_artifact:
            mock_artifact_instance = MagicMock()
            mock_artifact_instance.download.return_value = '/tmp/test'
            mock_artifact.return_value = mock_artifact_instance
            
            with patch('os.listdir', return_value=['checkpoint.pth']), \
                 patch('torch.load', return_value=sample_checkpoint_data):
                
                result = mock_wandb_logger.load_checkpoint(version="latest")

                assert result is not None
                assert result['epoch'] == 5
                assert 'model_state_dict' in result
                assert 'optimizer_state_dict' in result
                assert result['loss'] == 0.123

    def test_load_checkpoint_with_custom_artifact_name(self, mock_wandb_logger, sample_checkpoint_data):
        """Test loading checkpoint with custom artifact name."""
        custom_name = "custom_checkpoint"

        with patch.object(mock_wandb_logger.wandb_api, 'artifact') as mock_artifact:
            mock_artifact_instance = MagicMock()
            mock_artifact_instance.download.return_value = '/tmp/test'
            mock_artifact.return_value = mock_artifact_instance
            
            with patch('os.listdir', return_value=['checkpoint.pth']), \
                 patch('torch.load', return_value=sample_checkpoint_data):
                
                result = mock_wandb_logger.load_checkpoint(artifact_name=custom_name, version="v1")

                assert result is not None
                assert result['epoch'] == 5

    def test_load_checkpoint_defaults_to_checkpoint_name(self, mock_wandb_logger, sample_checkpoint_data):
        """Test that load_checkpoint defaults to get_checkpoint_name() when no artifact_name provided."""
        with patch.object(mock_wandb_logger.wandb_api, 'artifact') as mock_artifact:
            mock_artifact_instance = MagicMock()
            mock_artifact_instance.download.return_value = '/tmp/test'
            mock_artifact.return_value = mock_artifact_instance
            
            with patch('os.listdir', return_value=['checkpoint.pth']), \
                 patch('torch.load', return_value=sample_checkpoint_data):
                
                result = mock_wandb_logger.load_checkpoint()

                assert result is not None

    def test_load_checkpoint_handles_download_failure(self, mock_wandb_logger):
        """Test that load_checkpoint handles download failures gracefully."""
        with patch.object(mock_wandb_logger.wandb_api, 'artifact') as mock_artifact:
            mock_artifact.side_effect = Exception("Download failed")

            result = mock_wandb_logger.load_checkpoint()

            assert result is None

    def test_load_checkpoint_with_wandb_disabled(self):
        """Test that load_checkpoint raises RuntimeError when WandB is disabled."""
        from src.train.wandb_logger import WandbLogger
        logger = WandbLogger(enabled=False)

        with pytest.raises(RuntimeError, match="Wandb not enabled or available"):
            logger.load_checkpoint()

    def test_resume_training_from_checkpoint_integration(self, mock_wandb_logger, sample_checkpoint_data):
        """Test the complete resume_training_from_checkpoint workflow."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with patch.object(mock_wandb_logger, 'load_checkpoint') as mock_load:
            mock_load.return_value = sample_checkpoint_data

            start_epoch = mock_wandb_logger.resume_training_from_checkpoint(
                model=model,
                optimizer=optimizer,
                artifact_name="test_checkpoint",
                version="latest"
            )

            assert start_epoch == 6  # epoch 5 + 1
            mock_load.assert_called_once_with("test_checkpoint", "latest")

    def test_resume_training_from_checkpoint_no_checkpoint(self, mock_wandb_logger):
        """Test resume_training_from_checkpoint when no checkpoint is available."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with patch.object(mock_wandb_logger, 'load_checkpoint') as mock_load:
            mock_load.return_value = None

            start_epoch = mock_wandb_logger.resume_training_from_checkpoint(
                model=model,
                optimizer=optimizer
            )

            assert start_epoch == 1  # Default to epoch 1 when no checkpoint
            mock_load.assert_called_once()

    def test_resume_training_from_checkpoint_with_wandb_disabled(self):
        """Test resume_training_from_checkpoint when WandB is disabled."""
        from src.train.wandb_logger import WandbLogger
        logger = WandbLogger(enabled=False)

        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with pytest.raises(RuntimeError, match="Wandb not enabled or available"):
            logger.resume_training_from_checkpoint(
                model=model,
                optimizer=optimizer
            )

    def test_checkpoint_retrieval_with_test_prefix(self):
        """Test that checkpoint retrieval includes test prefix when in test mode."""
        from src.train.wandb_logger import WandbLogger

        # Mock the _is_test_run method to return True
        with patch('src.train.wandb_logger.WandbLogger._is_test_run', return_value=True):
            logger = WandbLogger(enabled=True)
            checkpoint_name = logger.get_checkpoint_name()

            assert checkpoint_name == "test-checkpoint"

    def test_load_checkpoint_preserves_data_integrity(self, mock_wandb_logger, sample_checkpoint_data):
        """Test that loaded checkpoint data maintains integrity."""
        original_loss = sample_checkpoint_data['loss']
        original_epoch = sample_checkpoint_data['epoch']

        with patch.object(mock_wandb_logger.wandb_api, 'artifact') as mock_artifact:
            mock_artifact_instance = MagicMock()
            mock_artifact_instance.download.return_value = '/tmp/test'
            mock_artifact.return_value = mock_artifact_instance
            
            with patch('os.listdir', return_value=['checkpoint.pth']), \
                 patch('torch.load', return_value=sample_checkpoint_data):
                
                result = mock_wandb_logger.load_checkpoint()

                assert result is not None
                assert result['loss'] == original_loss
                assert result['epoch'] == original_epoch
                assert 'model_state_dict' in result
                assert 'optimizer_state_dict' in result
                assert isinstance(result['model_state_dict'], dict)
                assert isinstance(result['optimizer_state_dict'], dict)

    def test_save_model_does_not_delete_checkpoints(self):
        """Test that saving a final model to registry does not delete checkpoints."""
        from src.train.wandb_logger import WandbLogger
        import torch.nn as nn
        
        logger = WandbLogger(enabled=True)
        
        # Mock a run and initialize properly  
        mock_run = MagicMock()
        mock_run.name = "production-model"
        mock_run.log_artifact = MagicMock(return_value=MagicMock())
        mock_run.link_artifact = MagicMock()
        logger.run = mock_run
        logger.initialized = True  # Mark as initialized
        
        # Create a simple model for testing
        model = nn.Linear(10, 1)
        
        # Mock the executor to capture cleanup calls
        mock_executor = MagicMock()
        mock_cleanup_future = MagicMock()
        mock_executor.submit = MagicMock(return_value=mock_cleanup_future)
        logger._executor = mock_executor
        logger._pending_futures = []
        
        # Save model to registry with full mocking to avoid WandB API calls
        with patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('torch.save') as mock_torch_save, \
             patch('os.path.exists', return_value=True), \
             patch('os.unlink'), \
             patch('wandb.Artifact') as mock_artifact_class:
            
            # Mock the artifact creation and methods
            mock_artifact = MagicMock()
            mock_artifact_class.return_value = mock_artifact
            
            mock_temp_file = MagicMock()
            mock_temp_file.name = '/tmp/test_model.pth'
            mock_temp_file.__enter__.return_value = mock_temp_file
            mock_temp.return_value = mock_temp_file
            
            logger.save_model_to_registry(
                model=model,
                collection_name="test-model",
                metadata={"test": True}
            )
        
        # Verify that executor was called for model cleanup only
        assert mock_executor.submit.call_count == 1  # Only one cleanup call
        
        # Verify the cleanup call was for models, not checkpoints
        cleanup_call = mock_executor.submit.call_args_list[0]
        cleanup_args = cleanup_call[0]  # positional args
        
        # Should be cleaning up models, not checkpoints  
        assert "test-model" in str(cleanup_args[1])  # collection name
        assert cleanup_args[2] == "model"  # artifact type should be "model" not "model_checkpoint"
        
        # Verify no checkpoint cleanup was scheduled
        for call in mock_executor.submit.call_args_list:
            args = call[0]
            assert args[2] != "model_checkpoint", "Checkpoint cleanup should not be called when saving final model"

    def test_explicit_checkpoint_cleanup_method(self):
        """Test that the explicit cleanup_checkpoints method works correctly."""
        from src.train.wandb_logger import WandbLogger
        
        logger = WandbLogger(enabled=True)
        
        # Mock a run and initialize properly  
        mock_run = MagicMock()
        mock_run.name = "test-cleanup"
        logger.run = mock_run
        logger.initialized = True  # Mark as initialized
        
        # Mock the executor
        mock_executor = MagicMock()
        mock_cleanup_future = MagicMock()
        mock_executor.submit = MagicMock(return_value=mock_cleanup_future)
        logger._executor = mock_executor
        logger._pending_futures = []
        
        # Call explicit checkpoint cleanup
        logger.cleanup_checkpoints(keep_latest=3)
        
        # Verify cleanup was called correctly
        assert mock_executor.submit.call_count == 1
        
        cleanup_call = mock_executor.submit.call_args_list[0]
        cleanup_args = cleanup_call[0]
        
        # Verify correct parameters
        assert "checkpoint-test-cleanup" in str(cleanup_args[1])  # checkpoint collection name
        assert cleanup_args[2] == "model_checkpoint"  # correct artifact type
        assert cleanup_args[3] == 3  # keep_latest parameter

    def test_checkpoint_and_model_naming_distinction(self):
        """Test that checkpoints and final models use different artifact types and naming."""
        from src.train.wandb_logger import WandbLogger
        import torch.nn as nn
        
        logger = WandbLogger(enabled=True)
        
        # Mock a run
        mock_run = MagicMock()
        mock_run.name = "distinction-test"
        logger.run = mock_run
        
        # Test checkpoint naming (gets test prefix since it's a test run)
        checkpoint_name = logger._get_checkpoint_name()
        # Run name "distinction-test" contains "test", so gets test prefix
        assert checkpoint_name == "test-checkpoint-distinction-test"
        
        # Test model naming (should be different)
        model_name = logger._get_artifact_name("my-model")
        assert model_name == "test-my-model"  # Also gets test prefix
        assert model_name != checkpoint_name
        
        # Verify they serve different purposes
        assert "checkpoint" in checkpoint_name  # For training resumption
        assert "checkpoint" not in model_name   # For production use
        
        # Test with production run name (no "test" or "wandb" in name)
        mock_run.name = "production-v1"
        logger.run = mock_run
        
        production_checkpoint = logger._get_checkpoint_name()
        production_model = logger._get_artifact_name("prod-model")
        
        assert production_checkpoint == "checkpoint-production-v1"  # No test prefix
        assert production_model == "prod-model"  # No test prefix
        assert production_checkpoint != production_model

    def test_production_model_names_get_forced_test_prefix_in_tests(self):
        """Test that production model names are forced to have test prefix when called from tests."""
        from src.train.wandb_logger import WandbLogger
        import torch.nn as nn
        
        logger = WandbLogger(enabled=True)
        
        # Mock a run with production name
        mock_run = MagicMock()
        mock_run.name = "production-model-v2"
        mock_run.log_artifact = MagicMock(return_value=MagicMock())
        mock_run.link_artifact = MagicMock()
        logger.run = mock_run
        logger.initialized = True
        
        model = nn.Linear(10, 1)
        
        # Mock the executor
        mock_executor = MagicMock()
        mock_cleanup_future = MagicMock()
        mock_executor.submit = MagicMock(return_value=mock_cleanup_future)
        logger._executor = mock_executor
        logger._pending_futures = []
        
        # Try to save with a production model name
        production_model_name = "train.siamesenet_dino"  # This is what we want to prevent!
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('torch.save') as mock_torch_save, \
             patch('os.path.exists', return_value=True), \
             patch('os.unlink'), \
             patch('wandb.Artifact') as mock_artifact_class:
            
            mock_artifact = MagicMock()
            mock_artifact_class.return_value = mock_artifact
            
            mock_temp_file = MagicMock()
            mock_temp_file.name = '/tmp/test_model.pth'
            mock_temp_file.__enter__.return_value = mock_temp_file
            mock_temp.return_value = mock_temp_file
            
            logger.save_model_to_registry(
                model=model,
                collection_name=production_model_name,  # Dangerous production name
                metadata={"test": True}
            )
        
        # Verify the artifact was created with a test-prefixed name
        mock_artifact_class.assert_called_once()
        call_args = mock_artifact_class.call_args[1]  # keyword args
        artifact_name = call_args['name']
        
        # Should be forced to test prefix even though run name looks like production
        assert artifact_name.startswith('test-'), f"Production model name '{production_model_name}' should be forced to test prefix, got: {artifact_name}"
        assert artifact_name == f"test-{production_model_name}", f"Expected 'test-{production_model_name}', got: {artifact_name}"

    def test_checkpoint_name_includes_test_prefix_when_run_name_contains_test(self):
        """Test that checkpoint names get 'test-' prefix when run name contains test indicators."""
        from src.train.wandb_logger import WandbLogger
        
        logger = WandbLogger(enabled=True)
        
        # Test with run name that contains "test"
        mock_run = MagicMock()
        mock_run.name = "test-experiment-v1"
        logger.run = mock_run
        
        checkpoint_name = logger.get_checkpoint_name()
        assert checkpoint_name == "test-checkpoint-test-experiment-v1"
        
        # Test with run name that contains "wandb"
        mock_run.name = "wandb-sweep-123"
        logger.run = mock_run
        
        checkpoint_name = logger.get_checkpoint_name()
        assert checkpoint_name == "test-checkpoint-wandb-sweep-123"

    def test_checkpoint_name_no_test_prefix_for_production_runs(self):
        """Test that checkpoint names don't get 'test-' prefix for production run names."""
        from src.train.wandb_logger import WandbLogger
        
        logger = WandbLogger(enabled=True)
        
        # Test with production run name
        mock_run = MagicMock()
        mock_run.name = "production-model-v1"
        logger.run = mock_run
        
        checkpoint_name = logger.get_checkpoint_name()
        assert checkpoint_name == "checkpoint-production-model-v1"
        
        # Test with baseline run name
        mock_run.name = "baseline-experiment"
        logger.run = mock_run
        
        checkpoint_name = logger.get_checkpoint_name()
        assert checkpoint_name == "checkpoint-baseline-experiment"

    def test_save_and_load_checkpoints_with_different_runs(self):
        """
        Integration test to verify different runs create unique checkpoints
        and don't overwrite each other.
        """
        from src.train.wandb_logger import WandbLogger
        
        # Create two separate logger instances for different runs
        logger1 = WandbLogger(enabled=True)
        logger2 = WandbLogger(enabled=True)
        
        # Mock different runs
        mock_run1 = MagicMock()
        mock_run1.name = "resnet-training"
        logger1.run = mock_run1
        
        mock_run2 = MagicMock()
        mock_run2.name = "dinov3-training"
        logger2.run = mock_run2
        
        # Verify checkpoint names are different
        checkpoint_name_1 = logger1.get_checkpoint_name()
        checkpoint_name_2 = logger2.get_checkpoint_name()
        
        assert checkpoint_name_1 == "checkpoint-resnet-training"
        assert checkpoint_name_2 == "checkpoint-dinov3-training"
        assert checkpoint_name_1 != checkpoint_name_2
        
        # Verify the internal _get_checkpoint_name method produces unique names
        assert logger1._get_checkpoint_name() != logger2._get_checkpoint_name()
        
        # Test with different run names
        mock_run1.name = "experiment-1"
        mock_run2.name = "experiment-2"
        
        assert logger1._get_checkpoint_name() == "checkpoint-experiment-1"
        assert logger2._get_checkpoint_name() == "checkpoint-experiment-2"
        assert logger1._get_checkpoint_name() != logger2._get_checkpoint_name()

    def test_checkpoint_naming_prevents_overwrites(self):
        """
        Test that the new checkpoint naming system prevents overwrites
        between different model architectures and runs.
        """
        from src.train.wandb_logger import WandbLogger
        
        # Simulate different training scenarios that previously caused conflicts
        scenarios = [
            {"run_name": "resnet50-baseline", "expected": "checkpoint-resnet50-baseline"},
            {"run_name": "dinov3-experiment", "expected": "checkpoint-dinov3-experiment"}, 
            {"run_name": "siamese-net-v1", "expected": "checkpoint-siamese-net-v1"},
            {"run_name": "training-run-abc", "expected": "checkpoint-training-run-abc"}
        ]
        
        checkpoint_names = []
        for scenario in scenarios:
            logger = WandbLogger(enabled=True)
            mock_run = MagicMock()
            mock_run.name = scenario["run_name"]
            logger.run = mock_run
            
            checkpoint_name = logger.get_checkpoint_name()
            assert checkpoint_name == scenario["expected"]
            checkpoint_names.append(checkpoint_name)
        
        # Verify all checkpoint names are unique
        assert len(checkpoint_names) == len(set(checkpoint_names)), "Checkpoint names should all be unique"
        
        # Verify no two runs would create the same checkpoint artifact
        for i, name1 in enumerate(checkpoint_names):
            for j, name2 in enumerate(checkpoint_names):
                if i != j:
                    assert name1 != name2, f"Checkpoint names should be unique: {name1} == {name2}"

    def test_load_checkpoint_uses_correct_artifact_name(self, mock_wandb_logger, sample_checkpoint_data):
        """Test that load_checkpoint uses the correct artifact name based on run name."""
        # Mock a run with specific name
        mock_run = MagicMock()
        mock_run.name = "specific-experiment"
        mock_wandb_logger.run = mock_run
        
        with patch.object(mock_wandb_logger.wandb_api, 'artifact') as mock_artifact:
            mock_artifact_instance = MagicMock()
            mock_artifact_instance.download.return_value = '/tmp/test'
            mock_artifact.return_value = mock_artifact_instance
            
            with patch('os.listdir', return_value=['checkpoint.pth']), \
                 patch('torch.load', return_value=sample_checkpoint_data):
                
                # Load checkpoint without specifying artifact_name (should use run-specific name)
                result = mock_wandb_logger.load_checkpoint()
                
                # Verify it tried to load the correct artifact
                expected_artifact_name = "checkpoint-specific-experiment"
                mock_artifact.assert_called_once()
                call_args = mock_artifact.call_args[0]
                assert expected_artifact_name in call_args[0]  # First argument should contain the artifact reference
