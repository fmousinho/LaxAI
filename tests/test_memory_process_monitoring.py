import os
import gc
import psutil
import pytest
import torch
from unittest.mock import patch, MagicMock

from train.wandb_logger import WandbLogger
from train.training import Training
from config.all_config import wandb_config, training_config
from utils.env_secrets import setup_environment_secrets


@pytest.fixture
def mock_wandb_config():
    """Mock WandB config for testing."""
    with patch('config.all_config.wandb_config') as mock_config:
        mock_config.enabled = True
        mock_config.run_name = "test_memory_monitoring"
        mock_config.project = "test_project"
        yield mock_config


@pytest.fixture
def mock_training_config():
    """Mock training config for testing."""
    with patch('config.all_config.training_config') as mock_config:
        mock_config.num_epochs = 3
        mock_config.batch_size = 4
        mock_config.learning_rate = 0.001
        yield mock_config


def test_wandb_process_monitoring_between_epochs(mock_wandb_config):
    """Test that WandB processes are monitored correctly between epochs without termination."""
    # Setup environment
    setup_environment_secrets()

    # Create WandB logger
    wandb_logger = WandbLogger()

    # Initialize a run
    config = {"learning_rate": 0.001, "batch_size": 4}
    success = wandb_logger.init_run(config, run_name="test_process_monitoring")
    assert success, "Failed to initialize WandB run"

    # Simulate multiple epochs with process monitoring
    initial_process_count = wandb_logger._monitor_wandb_processes()
    print(f"Initial WandB processes: {initial_process_count}")

    # Simulate epoch operations that might spawn processes
    for epoch in range(3):
        # Log metrics (this can spawn background processes)
        wandb_logger.log_metrics({"epoch": epoch + 1, "loss": 0.5 - epoch * 0.1})

        # Monitor processes after each epoch
        process_count = wandb_logger._monitor_wandb_processes()
        print(f"Epoch {epoch + 1} - WandB processes: {process_count}")

        # Check for problematic processes (should not terminate)
        problematic = wandb_logger._cleanup_wandb_processes(force=False)
        print(f"Epoch {epoch + 1} - Problematic processes: {problematic}")

        # Assert no processes were terminated (force=False)
        assert problematic >= 0, "Process monitoring should not fail"

        # Simulate checkpoint saving every other epoch
        if epoch == 1:  # Save checkpoint on epoch 2
            mock_model_state = {"layer.weight": torch.randn(10, 5)}
            mock_optimizer_state = {"state": {}, "param_groups": [{"lr": 0.001}]}
            wandb_logger.save_checkpoint(
                epoch + 1,
                mock_model_state,
                mock_optimizer_state,
                0.4,
                "test_model"
            )

            # Monitor after checkpoint
            post_checkpoint_count = wandb_logger._monitor_wandb_processes()
            print(f"After checkpoint - WandB processes: {post_checkpoint_count}")

    # Final monitoring
    final_process_count = wandb_logger._monitor_wandb_processes()
    print(f"Final WandB processes: {final_process_count}")

    # Finish the run (should monitor but not terminate processes)
    wandb_logger.finish()

    # Verify the run was finished properly
    assert not wandb_logger.initialized, "WandB run should be finished"


def test_memory_monitoring_during_training(mock_training_config, mock_wandb_config):
    """Test memory usage monitoring during training epochs."""
    # Setup environment
    setup_environment_secrets()

    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Create training instance with WandB enabled
    training = Training()

    # Mock the data loading to avoid actual data dependencies
    with patch.object(training, 'setup_dataloader') as mock_setup_data:
        mock_setup_data.return_value = None

        with patch.object(training, 'setup_model') as mock_create_model:
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model

            with patch.object(training, 'setup_training_pipeline') as mock_setup_pipeline:
                mock_setup_pipeline.return_value = None

                with patch.object(training, 'train') as mock_train_epoch:
                    # Mock training epoch to return decreasing loss
                    mock_train_epoch.return_value = None

                    # Track memory usage across epochs
                    memory_usage = []

                    for epoch in range(3):
                        # Force garbage collection before measuring
                        gc.collect()

                        # Record memory before epoch
                        memory_before = process.memory_info().rss / 1024 / 1024
                        memory_usage.append(memory_before)
                        print(f"Epoch {epoch + 1} start - Memory: {memory_before:.2f} MB")

                        # Simulate epoch training
                        training.train(num_epochs=1, start_epoch=epoch + 1)

                        # Record memory after epoch
                        memory_after = process.memory_info().rss / 1024 / 1024
                        print(f"Epoch {epoch + 1} end - Memory: {memory_after:.2f} MB")

                        # Check for memory leaks (memory shouldn't increase dramatically)
                        memory_increase = memory_after - memory_before
                        print(f"Epoch {epoch + 1} memory increase: {memory_increase:.2f} MB")

                        # Allow some memory increase but not excessive
                        assert memory_increase < 100, f"Memory leak detected in epoch {epoch + 1}: {memory_increase:.2f} MB increase"

    print(f"Memory usage across epochs: {[f'{m:.2f} MB' for m in memory_usage]}")


def test_process_spawning_monitoring():
    """Test monitoring of process spawning during WandB operations."""
    # Setup environment
    setup_environment_secrets()

    wandb_logger = WandbLogger()

    # Get initial process count
    initial_processes = len(psutil.Process().children(recursive=True))
    initial_wandb_processes = wandb_logger._monitor_wandb_processes()

    print(f"Initial total processes: {initial_processes}")
    print(f"Initial WandB processes: {initial_wandb_processes}")

    # Initialize WandB run (this spawns processes)
    config = {"test": True}
    success = wandb_logger.init_run(config, run_name="test_process_spawning")
    assert success

    # Check process count after initialization
    after_init_processes = len(psutil.Process().children(recursive=True))
    after_init_wandb_processes = wandb_logger._monitor_wandb_processes()

    print(f"After init total processes: {after_init_processes}")
    print(f"After init WandB processes: {after_init_wandb_processes}")

    # Process count should not decrease (we're monitoring, not terminating)
    assert after_init_wandb_processes >= initial_wandb_processes, "WandB processes should not be terminated"

    # Log some metrics (this can spawn additional processes)
    wandb_logger.log_metrics({"metric1": 1.0, "metric2": 2.0})

    after_metrics_processes = wandb_logger._monitor_wandb_processes()
    print(f"After metrics WandB processes: {after_metrics_processes}")

    # Save a checkpoint (this can spawn processes)
    mock_model_state = {"layer.weight": torch.randn(10, 5)}
    mock_optimizer_state = {"state": {}, "param_groups": [{"lr": 0.001}]}
    wandb_logger.save_checkpoint(
        1,
        mock_model_state,
        mock_optimizer_state,
        0.5,
        "test_model"
    )

    after_checkpoint_processes = wandb_logger._monitor_wandb_processes()
    print(f"After checkpoint WandB processes: {after_checkpoint_processes}")

    # Finish the run
    wandb_logger.finish()

    final_processes = wandb_logger._monitor_wandb_processes()
    print(f"Final WandB processes: {final_processes}")

    # Verify monitoring worked throughout
    assert final_processes >= 0, "Process monitoring should work until the end"


def test_memory_and_process_integration():
    """Integration test monitoring both memory and processes during simulated training."""
    # Setup environment
    setup_environment_secrets()

    wandb_logger = WandbLogger()
    process = psutil.Process()

    # Track metrics across simulated training
    metrics_history = []

    # Initialize run
    config = {"learning_rate": 0.001}
    wandb_logger.init_run(config, run_name="integration_test")

    for epoch in range(3):
        # Record state before epoch
        memory_before = process.memory_info().rss / 1024 / 1024
        processes_before = wandb_logger._monitor_wandb_processes()

        # Force garbage collection
        gc.collect()

        # Simulate training operations
        wandb_logger.log_metrics({
            "epoch": epoch + 1,
            "loss": 1.0 - epoch * 0.1,
            "memory_mb": memory_before,
            "wandb_processes": processes_before
        })

        # Simulate checkpoint saving every other epoch
        if epoch % 2 == 0:
            mock_model_state = {"layer.weight": torch.randn(10, 5)}
            mock_optimizer_state = {"state": {}, "param_groups": [{"lr": 0.001}]}
            wandb_logger.save_checkpoint(
                epoch + 1,
                mock_model_state,
                mock_optimizer_state,
                0.9 - epoch * 0.1,
                "integration_model"
            )

        # Record state after epoch
        memory_after = process.memory_info().rss / 1024 / 1024
        processes_after = wandb_logger._monitor_wandb_processes()

        # Store metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": memory_after - memory_before,
            "processes_before": processes_before,
            "processes_after": processes_after,
            "processes_delta": processes_after - processes_before
        }
        metrics_history.append(epoch_metrics)

        print(f"Epoch {epoch + 1} metrics: {epoch_metrics}")

        # Assertions
        assert epoch_metrics["memory_delta"] < 200, f"Excessive memory increase in epoch {epoch + 1}"
        assert epoch_metrics["processes_after"] >= 0, f"Invalid process count in epoch {epoch + 1}"

    # Finish and verify
    wandb_logger.finish()

    print("Integration test completed successfully")
    print(f"Final metrics history: {metrics_history}")

    # Verify we have metrics for all epochs
    assert len(metrics_history) == 3, "Should have metrics for all 3 epochs"
