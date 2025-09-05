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
        assert checkpoint_name == "checkpoint"

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
