import pytest
import torch
import gc
import time
import psutil
from tests.fixtures.wandb_test_helpers import wandb_test_manager, create_minimal_checkpoint
from train.wandb_logger import WandbLogger
from utils.env_secrets import setup_environment_secrets


@pytest.mark.integration
def test_memory_monitoring_with_cleanup(wandb_test_manager):
    """Memory monitoring test with proper WandB artifact cleanup."""
    setup_environment_secrets()
    
    wandb_logger = WandbLogger()
    process = psutil.Process()
    
    # Initialize run
    config = {"learning_rate": 0.001}
    wandb_logger.init_run(config, run_name="memory_test_improved")
    
    # Track the checkpoint name for cleanup
    checkpoint_name = wandb_logger._sanitize_artifact_name(wandb_logger._get_checkpoint_name())
    wandb_test_manager.track_artifact(checkpoint_name)
    
    try:
        # Allow WandB initialization to settle
        gc.collect()
        time.sleep(0.5)

        baseline_memory = process.memory_info().rss / 1024 / 1024
        print(f"Baseline memory: {baseline_memory:.2f} MB")

        for epoch in range(2):  # Use max 2 epochs for tests
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Log minimal metrics
            wandb_logger.log_metrics({
                "epoch": epoch + 1,
                "loss": 1.0 - epoch * 0.1,
                "memory_mb": memory_before
            })

            # Save minimal checkpoints
            if epoch % 2 == 0:
                checkpoint_data = create_minimal_checkpoint(epoch + 1)
                wandb_logger.save_checkpoint(**checkpoint_data)

            # Force garbage collection and measure memory
            gc.collect()
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before

            print(f"Epoch {epoch + 1}: {memory_before:.2f} MB -> {memory_after:.2f} MB (Δ{memory_delta:.2f} MB)")

            # Memory delta should be reasonable (much less than 50MB for minimal checkpoints)
            assert memory_delta < 10, f"Excessive memory increase in epoch {epoch + 1}: {memory_delta:.2f} MB"

    finally:
        # Always finish the run
        try:
            wandb_logger.finish()
        except Exception:
            pass


@pytest.mark.integration
def test_process_monitoring_lightweight(wandb_test_manager):
    """Lightweight process monitoring test."""
    setup_environment_secrets()
    
    wandb_logger = WandbLogger()
    
    # Initialize run
    wandb_logger.init_run({"test": True}, run_name="process_test_lightweight")
    
    # Track for cleanup
    checkpoint_name = wandb_logger._sanitize_artifact_name(wandb_logger._get_checkpoint_name())
    wandb_test_manager.track_artifact(checkpoint_name)
    
    try:
        initial_processes = wandb_logger._monitor_wandb_processes()
        print(f"Initial WandB processes: {initial_processes}")
        
        # Simulate some work with minimal checkpointing
        for epoch in range(2):  # Use max 2 epochs for tests
            wandb_logger.log_metrics({"epoch": epoch + 1, "test_metric": 0.5})
            
            if epoch == 1:  # Only one checkpoint to minimize artifacts
                checkpoint_data = create_minimal_checkpoint(epoch + 1)
                wandb_logger.save_checkpoint(**checkpoint_data)
                
        final_processes = wandb_logger._monitor_wandb_processes()
        print(f"Final WandB processes: {final_processes}")
        
        assert final_processes >= 0, "Process monitoring should work"
        
    finally:
        try:
            wandb_logger.finish()
        except Exception:
            pass


@pytest.mark.integration
def test_memory_process_integration_minimal(wandb_test_manager):
    """Minimal integration test for memory and process monitoring."""
    setup_environment_secrets()
    
    wandb_logger = WandbLogger()
    process = psutil.Process()
    
    config = {"learning_rate": 0.001}
    wandb_logger.init_run(config, run_name="integration_minimal")
    
    # Track for cleanup
    checkpoint_name = wandb_logger._sanitize_artifact_name(wandb_logger._get_checkpoint_name())
    wandb_test_manager.track_artifact(checkpoint_name)
    
    try:
        # Allow initialization to settle
        gc.collect()
        time.sleep(0.5)
        
        # Single epoch test to minimize time and artifacts
        memory_before = process.memory_info().rss / 1024 / 1024
        processes_before = wandb_logger._monitor_wandb_processes()
        
        # Minimal work
        wandb_logger.log_metrics({
            "epoch": 1,
            "loss": 0.5,
            "memory_mb": memory_before,
            "wandb_processes": processes_before
        })
        
        # One minimal checkpoint
        checkpoint_data = create_minimal_checkpoint(1)
        wandb_logger.save_checkpoint(**checkpoint_data)
        
        # Check final state
        gc.collect()
        memory_after = process.memory_info().rss / 1024 / 1024
        processes_after = wandb_logger._monitor_wandb_processes()
        
        memory_delta = memory_after - memory_before
        
        print(f"Memory: {memory_before:.2f} -> {memory_after:.2f} MB (Δ{memory_delta:.2f} MB)")
        print(f"Processes: {processes_before} -> {processes_after}")
        
        # Assertions with reasonable limits for minimal data
        assert memory_delta < 5, f"Excessive memory increase: {memory_delta:.2f} MB"
        assert processes_after >= 0, "Process monitoring failed"
        
    finally:
        try:
            wandb_logger.finish()
        except Exception:
            pass
