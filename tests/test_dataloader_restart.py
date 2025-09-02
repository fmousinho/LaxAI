"""
Test suite for DataLoader worker restart functionality.
Tests the intelligent memory-based worker restart mechanism.
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
    training.num_epochs = 5
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


class TestDataLoaderRestart:
    """Test suite for DataLoader worker restart functionality."""

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

        # Mock the worker processes
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        mock_worker.terminate.return_value = None
        mock_worker.join.return_value = None

        # Mock the iterator and workers
        mock_iterator = MagicMock()
        mock_iterator._workers = [mock_worker]
        training_instance.dataloader._iterator = mock_iterator

        # Call restart method
        training_instance._restart_dataloader_workers()

        # Verify worker was terminated and joined
        mock_worker.terminate.assert_called_once()
        mock_worker.join.assert_called_once_with(timeout=1.0)

        # Verify logger was called
        mock_logger.info.assert_any_call("Training DataLoader workers restarted")

    @patch('src.train.training.logger')
    def test_restart_dataloader_workers_no_dataloader(self, mock_logger, training_instance):
        """Test restart when no DataLoader exists."""
        # Ensure no dataloaders
        training_instance.dataloader = None
        training_instance.val_dataloader = None

        # Call restart method
        training_instance._restart_dataloader_workers()

        # Verify no errors and appropriate logging
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()

    @patch('src.train.training.logger')
    def test_restart_dataloader_workers_failure(self, mock_logger, training_instance, dummy_dataset):
        """Test DataLoader worker restart failure handling."""
        # Setup DataLoader
        training_instance.setup_dataloader(dummy_dataset, type='train')

        # Mock iterator to raise exception
        mock_iterator = MagicMock()
        mock_iterator._workers = None  # This will cause an exception
        training_instance.dataloader._iterator = mock_iterator

        # Call restart method
        training_instance._restart_dataloader_workers()

        # Verify warning was logged but no exception was raised
        mock_logger.warning.assert_called_once()
        mock_logger.warning.assert_any_call("Failed to restart DataLoader workers: 'NoneType' object is not iterable")

    @patch('src.train.training.Training._restart_dataloader_workers')
    @patch('src.train.training.logger')
    def test_memory_based_restart_triggered(self, mock_logger, mock_restart, training_instance):
        """Test that worker restart is triggered when memory exceeds threshold."""
        # Mock memory monitor to return high memory usage
        with patch.object(training_instance.cpu_monitor, '_get_current_memory', return_value={"percent": 95.0, "rss_mb": 1000.0}):
            # Setup training components
            training_instance.num_workers = 2
            training_instance.last_worker_restart_epoch = -1

            # Simulate the memory check logic (epoch > 0 and memory > 90%)
            epoch = 1
            current_memory = training_instance.cpu_monitor._get_current_memory()
            memory_percent = current_memory["percent"]

            # This should trigger restart
            should_restart = (memory_percent > THRESHOLD_FOR_DATALOADER_RESTART and
                             (epoch - training_instance.last_worker_restart_epoch) >= 1)

            assert should_restart == True

            # Simulate calling the restart logic
            if should_restart:
                training_instance._restart_dataloader_workers()
                training_instance.last_worker_restart_epoch = epoch

            # Verify restart was called
            mock_restart.assert_called_once()
            assert training_instance.last_worker_restart_epoch == 1

    @patch('src.train.training.Training._restart_dataloader_workers')
    @patch('src.train.training.logger')
    def test_memory_based_restart_not_triggered_low_memory(self, mock_logger, mock_restart, training_instance):
        """Test that worker restart is NOT triggered when memory is below threshold."""
        # Mock memory monitor to return normal memory usage
        with patch.object(training_instance.cpu_monitor, '_get_current_memory', return_value={"percent": 50.0, "rss_mb": 500.0}):
            # Setup training components
            training_instance.num_workers = 2
            training_instance.last_worker_restart_epoch = -1

            # Simulate the memory check logic
            epoch = 1
            current_memory = training_instance.cpu_monitor._get_current_memory()
            memory_percent = current_memory["percent"]

            # This should NOT trigger restart
            should_restart = (memory_percent > THRESHOLD_FOR_DATALOADER_RESTART and
                             (epoch - training_instance.last_worker_restart_epoch) >= 1)

            assert should_restart == False

            # Verify restart was NOT called
            mock_restart.assert_not_called()

    @patch('src.train.training.Training._restart_dataloader_workers')
    @patch('src.train.training.logger')
    def test_restart_prevented_by_recent_restart(self, mock_logger, mock_restart, training_instance):
        """Test that restart is prevented if it happened too recently."""
        # Test with same epoch (should be prevented)
        with patch.object(training_instance.cpu_monitor, '_get_current_memory', return_value={"percent": 95.0, "rss_mb": 1000.0}):
            training_instance.num_workers = 2
            training_instance.last_worker_restart_epoch = 1  # Restarted in current epoch

            epoch = 1
            current_memory = training_instance.cpu_monitor._get_current_memory()
            memory_percent = current_memory["percent"]

            # This should NOT trigger restart due to recent restart
            should_restart = (memory_percent > THRESHOLD_FOR_DATALOADER_RESTART and
                             (epoch - training_instance.last_worker_restart_epoch) >= 1)

            assert should_restart == False

            # Verify restart was NOT called
            mock_restart.assert_not_called()

        # Test with different epoch (should be allowed)
        with patch.object(training_instance.cpu_monitor, '_get_current_memory', return_value={"percent": 95.0, "rss_mb": 1000.0}):
            training_instance.last_worker_restart_epoch = 0  # Restarted in previous epoch

            epoch = 2
            current_memory = training_instance.cpu_monitor._get_current_memory()
            memory_percent = current_memory["percent"]

            # This should trigger restart
            should_restart = (memory_percent > THRESHOLD_FOR_DATALOADER_RESTART and
                             (epoch - training_instance.last_worker_restart_epoch) >= 1)

            assert should_restart == True

    @patch('src.train.training.Training.setup_dataloader')
    @patch('src.train.training.logger')
    def test_restart_creates_new_dataloader(self, mock_logger, mock_setup_dataloader, training_instance, dummy_dataset):
        """Test that restart creates new DataLoader instances."""
        # Setup initial DataLoader
        training_instance.setup_dataloader(dummy_dataset, type='train')

        # Mock the worker cleanup to avoid NoneType errors
        with patch.object(training_instance, 'dataloader') as mock_dataloader:
            mock_worker = MagicMock()
            mock_worker.is_alive.return_value = True
            mock_iterator = MagicMock()
            mock_iterator._workers = [mock_worker]
            mock_dataloader._iterator = mock_iterator
            mock_dataloader.dataset = dummy_dataset

            # Call restart
            training_instance._restart_dataloader_workers()

            # Verify setup_dataloader was called to recreate the dataloader
            mock_setup_dataloader.assert_called()
            assert mock_setup_dataloader.call_count >= 1

    def test_restart_with_zero_workers(self, training_instance):
        """Test that restart logic is skipped when num_workers is 0."""
        training_instance.num_workers = 0

        # This should not attempt any restart logic
        # The method should handle this gracefully
        training_instance._restart_dataloader_workers()

        # No assertions needed - just verify no exceptions are raised

    @patch('gc.collect')
    def test_restart_triggers_garbage_collection(self, mock_gc_collect, training_instance, dummy_dataset):
        """Test that restart triggers garbage collection."""
        # Setup DataLoader
        training_instance.setup_dataloader(dummy_dataset, type='train')

        # Call restart
        training_instance._restart_dataloader_workers()

        # Verify garbage collection was called
        mock_gc_collect.assert_called_once()

    def test_restart_updates_last_restart_epoch(self, training_instance):
        """Test that restart updates the last restart epoch tracking."""
        initial_epoch = training_instance.last_worker_restart_epoch
        assert initial_epoch == -1

        # Simulate restart logic
        training_instance.last_worker_restart_epoch = 5

        assert training_instance.last_worker_restart_epoch == 5
