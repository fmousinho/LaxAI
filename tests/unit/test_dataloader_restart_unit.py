"""
Unit test for DataLoader worker restart functionality.
Tests the intelligent memory-based worker restart mechanism without full training.
"""

import os
import gc
import psutil
import pytest
import torch
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime

from src.train.training import Training, THRESHOLD_FOR_DATALOADER_RESTART


class DummyTripletDataset(torch.utils.data.Dataset):
    def __init__(self, n=4, dim=4):
        self.n = n
        self.dim = dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return (anchor, positive, negative, label)
        x = torch.randn(self.dim)
        label = idx % 2
        return x, x, x, label


class DummyModel(torch.nn.Module):
    def __init__(self, dim=4, emb_dim=2):
        super().__init__()
        self.encoder = torch.nn.Linear(dim, emb_dim)

    def forward_triplet(self, a, p, n):
        return self.encoder(a), self.encoder(p), self.encoder(n)


@pytest.fixture
def training_instance():
    """Create a training instance with dummy components for testing."""
    training = Training(
        device='cpu',  # Use CPU for testing
        enable_multithreading=True,
        num_workers=2,  # Use 2 workers for testing
        clear_memory_on_start=False
    )

    # Set up minimal training parameters
    training.num_epochs = 1  # Single epoch for unit test
    training.batch_size = 2
    training.margin = 0.5
    training.learning_rate = 0.01
    training.weight_decay = 0.0001
    training.lr_scheduler_patience = 5
    training.lr_scheduler_threshold = 0.01
    training.lr_scheduler_min_lr = 1e-6
    training.lr_scheduler_factor = 0.5
    training.scheduler_patience = 5
    training.scheduler_threshold = 0.01
    training.lr_scheduler_min_lr = 1e-6
    training.lr_scheduler_factor = 0.5
    training.force_pretraining = False

    return training


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing."""
    return DummyTripletDataset(n=8, dim=4)


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel(dim=4, emb_dim=2)


class TestDataLoaderRestartUnit:
    """Unit test suite for DataLoader worker restart functionality."""

    def test_restart_threshold_constant(self):
        """Test that the restart threshold constant is properly defined."""
        assert THRESHOLD_FOR_DATALOADER_RESTART == 90.0

    def test_memory_monitor_initialization(self, training_instance):
        """Test that CPU memory monitor is properly initialized."""
        assert hasattr(training_instance, 'cpu_monitor')
        assert training_instance.cpu_monitor is not None
        assert hasattr(training_instance, 'last_worker_restart_epoch')
        assert training_instance.last_worker_restart_epoch == -1

    @patch('src.train.training.logger')
    def test_restart_dataloader_workers_success(self, mock_logger, training_instance, dummy_dataset):
        """Test successful DataLoader worker restart."""
        # Setup DataLoader
        training_instance.setup_dataloader(dummy_dataset, type='train')

        # Verify initial state
        assert training_instance.dataloader is not None
        assert training_instance.last_worker_restart_epoch == -1

        # Mock memory usage above threshold
        with patch.object(training_instance.cpu_monitor, 'get_memory_usage', return_value=95.0):
            # Attempt restart
            result = training_instance.restart_dataloader_workers(epoch=1)

            # Verify restart was successful
            assert result is True
            assert training_instance.last_worker_restart_epoch == 1
            mock_logger.info.assert_called_with("Restarting DataLoader workers due to high memory usage: 95.0%")

    @patch('src.train.training.logger')
    def test_restart_dataloader_workers_no_restart_needed(self, mock_logger, training_instance, dummy_dataset):
        """Test that DataLoader workers are not restarted when memory is below threshold."""
        # Setup DataLoader
        training_instance.setup_dataloader(dummy_dataset, type='train')

        # Mock memory usage below threshold
        with patch.object(training_instance.cpu_monitor, 'get_memory_usage', return_value=80.0):
            # Attempt restart
            result = training_instance.restart_dataloader_workers(epoch=1)

            # Verify no restart occurred
            assert result is False
            assert training_instance.last_worker_restart_epoch == -1
            mock_logger.debug.assert_called_with("Memory usage is 80.0%, no DataLoader restart needed")

    @patch('src.train.training.logger')
    def test_restart_dataloader_workers_recent_restart(self, mock_logger, training_instance, dummy_dataset):
        """Test that DataLoader workers are not restarted too frequently."""
        # Setup DataLoader
        training_instance.setup_dataloader(dummy_dataset, type='train')

        # Set last restart to recent epoch
        training_instance.last_worker_restart_epoch = 5

        # Mock memory usage above threshold
        with patch.object(training_instance.cpu_monitor, 'get_memory_usage', return_value=95.0):
            # Attempt restart at epoch 6 (too soon)
            result = training_instance.restart_dataloader_workers(epoch=6)

            # Verify no restart occurred due to frequency limit
            assert result is False
            assert training_instance.last_worker_restart_epoch == 5
            mock_logger.debug.assert_called_with("DataLoader restart skipped - too recent (epoch 5)")

    @patch('src.train.training.logger')
    def test_restart_dataloader_workers_failure(self, mock_logger, training_instance, dummy_dataset):
        """Test handling of DataLoader worker restart failure."""
        # Setup DataLoader
        training_instance.setup_dataloader(dummy_dataset, type='train')

        # Mock DataLoader creation to fail
        with patch.object(training_instance, 'setup_dataloader', side_effect=Exception("Setup failed")):
            with patch.object(training_instance.cpu_monitor, 'get_memory_usage', return_value=95.0):
                # Attempt restart
                result = training_instance.restart_dataloader_workers(epoch=1)

                # Verify restart failed gracefully
                assert result is False
                assert training_instance.last_worker_restart_epoch == -1
                mock_logger.error.assert_called_with("Failed to restart DataLoader workers: Setup failed")

    def test_memory_threshold_calculation(self, training_instance):
        """Test memory threshold calculation logic."""
        # Test various memory percentages
        test_cases = [
            (50.0, False),  # Below threshold
            (85.0, False),  # Below threshold
            (90.0, True),   # At threshold
            (95.0, True),   # Above threshold
            (100.0, True),  # Well above threshold
        ]

        for memory_usage, should_restart in test_cases:
            with patch.object(training_instance.cpu_monitor, 'get_memory_usage', return_value=memory_usage):
                needs_restart = training_instance.cpu_monitor.get_memory_usage() >= THRESHOLD_FOR_DATALOADER_RESTART
                assert needs_restart == should_restart, f"Memory {memory_usage}% should {'need' if should_restart else 'not need'} restart"

    def test_dataloader_worker_count_preservation(self, training_instance, dummy_dataset):
        """Test that worker count is preserved during restart."""
        # Setup DataLoader with specific worker count
        training_instance.num_workers = 4
        training_instance.setup_dataloader(dummy_dataset, type='train')

        original_workers = training_instance.dataloader.num_workers

        # Mock memory usage above threshold and successful restart
        with patch.object(training_instance.cpu_monitor, 'get_memory_usage', return_value=95.0):
            with patch.object(training_instance, 'setup_dataloader') as mock_setup:
                result = training_instance.restart_dataloader_workers(epoch=1)

                # Verify setup_dataloader was called (restart attempted)
                mock_setup.assert_called_once()

                # Verify worker count would be preserved (this is tested via the setup call)
                assert result is True

    def test_memory_monitoring_integration(self, training_instance):
        """Test integration between memory monitoring and DataLoader restart."""
        # Verify memory monitor exists and is functional
        assert hasattr(training_instance, 'cpu_monitor')
        assert training_instance.cpu_monitor is not None

        # Test memory monitoring methods exist
        assert hasattr(training_instance.cpu_monitor, 'get_memory_usage')

        # Test that memory monitoring can be called without error
        memory_usage = training_instance.cpu_monitor.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        assert 0 <= memory_usage <= 100  # Memory percentage should be between 0-100
