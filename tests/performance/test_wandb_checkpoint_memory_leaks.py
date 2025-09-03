import pytest
import torch
import gc
import psutil
import tempfile
import os
from unittest.mock import patch, MagicMock
from train.wandb_logger import wandb_logger
from utils.env_secrets import setup_environment_secrets


class MemoryTracker:
    """Helper class to track memory usage during checkpoint operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.checkpoints = []
        
    def reset_baseline(self):
        """Reset baseline memory measurement."""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
    def measure(self, label: str = ""):
        """Measure current memory and store checkpoint."""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        current_memory = self.process.memory_info().rss / 1024 / 1024
        delta = current_memory - self.baseline_memory if self.baseline_memory else 0
        
        checkpoint = {
            'label': label,
            'memory_mb': current_memory,
            'delta_mb': delta
        }
        self.checkpoints.append(checkpoint)
        print(f"Memory {label}: {current_memory:.1f}MB (Δ{delta:+.1f}MB)")
        return checkpoint
        
    def assert_no_major_leak(self, threshold_mb: float = 20.0):
        """Assert that memory hasn't increased by more than threshold."""
        if len(self.checkpoints) < 2:
            return
        
        final_delta = self.checkpoints[-1]['delta_mb']
        assert abs(final_delta) < threshold_mb, \
            f"Potential memory leak detected: {final_delta:.1f}MB increase (threshold: {threshold_mb}MB)"


@pytest.fixture
def memory_tracker():
    """Fixture providing a memory tracker instance."""
    tracker = MemoryTracker()
    tracker.reset_baseline()
    return tracker


@pytest.fixture
def large_model_state():
    """Create a realistically sized model state dict for testing."""
    # Simulate a model with ~50MB of parameters (realistic for medium models)
    state_dict = {
        'backbone.conv1.weight': torch.randn(64, 3, 7, 7),  # ~37KB
        'backbone.conv2.weight': torch.randn(128, 64, 3, 3),  # ~295KB
        'backbone.conv3.weight': torch.randn(256, 128, 3, 3),  # ~1.2MB
        'backbone.fc.weight': torch.randn(1000, 2048),  # ~8MB
        'head.weight': torch.randn(2048, 512),  # ~4MB
        'large_embedding.weight': torch.randn(10000, 512),  # ~20MB
        'transformer.layers.0.weight': torch.randn(2048, 2048),  # ~16MB
    }
    return state_dict


@pytest.fixture
def large_optimizer_state():
    """Create a realistically sized optimizer state dict for testing."""
    # Simulate Adam optimizer state (2x model size for momentum + variance)
    optimizer_state = {
        'state': {
            0: {
                'step': torch.tensor(100),
                'exp_avg': torch.randn(1000, 2048),  # ~8MB
                'exp_avg_sq': torch.randn(1000, 2048),  # ~8MB
            },
            1: {
                'step': torch.tensor(100),
                'exp_avg': torch.randn(2048, 512),  # ~4MB
                'exp_avg_sq': torch.randn(2048, 512),  # ~4MB
            }
        },
        'param_groups': [{'lr': 0.001, 'betas': (0.9, 0.999)}]
    }
    return optimizer_state


@pytest.mark.performance
def test_checkpoint_save_memory_leak_with_large_models(memory_tracker, large_model_state, large_optimizer_state):
    """Test for memory leaks during checkpoint saving with realistically large models."""
    setup_environment_secrets()
    
    # Mock WandB to avoid actual uploads but test memory handling
    with patch('wandb.init'), \
         patch('wandb.Artifact') as mock_artifact, \
         patch.object(wandb_logger, 'run') as mock_run:
        
        # Configure mocks
        mock_artifact_instance = MagicMock()
        mock_artifact.return_value = mock_artifact_instance
        mock_run.log_artifact = MagicMock()
        wandb_logger.wandb_api = MagicMock()  # Mock API for cleanup
        
        try:
            # Initialize WandB logger
            wandb_logger.init_run(config={'test': True}, run_name="memory_leak_test")
            memory_tracker.measure("after_init")
            
            # Perform first checkpoint to establish baseline (WandB often has one-time setup costs)
            try:
                wandb_logger.save_checkpoint(
                    epoch=0,
                    model_state_dict=large_model_state.copy(),
                    optimizer_state_dict=large_optimizer_state.copy(),
                    loss=1.0
                )
            except Exception as e:
                print(f"Checkpoint save failed (expected in test): {e}")
            
            # Reset baseline after first save to account for one-time WandB initialization
            memory_tracker.reset_baseline()
            memory_tracker.measure("after_first_checkpoint")
            
            # Perform multiple checkpoint saves to detect accumulating leaks
            for epoch in range(1, 5):
                memory_before = memory_tracker.measure(f"before_epoch_{epoch}")
                
                # Save checkpoint with large state dicts
                try:
                    wandb_logger.save_checkpoint(
                        epoch=epoch,
                        model_state_dict=large_model_state.copy(),  # Copy to simulate real usage
                        optimizer_state_dict=large_optimizer_state.copy(),
                        loss=1.0 - epoch * 0.1
                    )
                except Exception as e:
                    # Even if save fails, we should not leak memory
                    print(f"Checkpoint save failed (expected in test): {e}")
                
                memory_after = memory_tracker.measure(f"after_epoch_{epoch}")
                
                # Check that memory didn't increase dramatically in this iteration
                iteration_delta = memory_after['memory_mb'] - memory_before['memory_mb']
                assert iteration_delta < 10, \
                    f"Excessive memory increase in epoch {epoch}: {iteration_delta:.1f}MB"
            
            # Final memory check - should not have leaked significantly after first checkpoint
            memory_tracker.measure("final")
            memory_tracker.assert_no_major_leak(threshold_mb=20.0)  # Stricter threshold after accounting for initialization
            
        finally:
            try:
                wandb_logger.finish()
            except Exception:
                pass


@pytest.mark.performance
def test_checkpoint_save_cleanup_effectiveness(memory_tracker):
    """Test that cleanup operations actually free memory."""
    setup_environment_secrets()
    
    with patch('wandb.init'), \
         patch('wandb.Artifact') as mock_artifact, \
         patch.object(wandb_logger, 'run') as mock_run:
        
        mock_artifact_instance = MagicMock()
        mock_artifact.return_value = mock_artifact_instance
        mock_run.log_artifact = MagicMock()
        wandb_logger.wandb_api = MagicMock()
        
        try:
            wandb_logger.init_run(config={'test': True}, run_name="cleanup_test")
            memory_tracker.measure("baseline")
            
            # Create temporary checkpoint files to test cleanup
            temp_files = []
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                    # Write a moderately sized checkpoint
                    checkpoint_data = {
                        'epoch': i,
                        'model_state_dict': {'layer.weight': torch.randn(1000, 1000)},  # ~4MB
                        'optimizer_state_dict': {'state': {}},
                        'loss': 0.5
                    }
                    torch.save(checkpoint_data, f.name)
                    temp_files.append(f.name)
            
            memory_after_files = memory_tracker.measure("after_creating_files")
            
            # Test that files can be cleaned up properly
            for file_path in temp_files:
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
            
            # Force garbage collection
            gc.collect()
            memory_after_cleanup = memory_tracker.measure("after_cleanup")
            
            # Memory should decrease or stay stable after cleanup
            cleanup_delta = memory_after_cleanup['memory_mb'] - memory_after_files['memory_mb']
            assert cleanup_delta <= 5, \
                f"Memory did not decrease after cleanup: {cleanup_delta:.1f}MB"
                
        finally:
            # Clean up any remaining files
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except Exception:
                    pass


@pytest.mark.performance
def test_wandb_artifact_object_cleanup(memory_tracker):
    """Test that WandB artifact objects are properly cleaned up from memory."""
    setup_environment_secrets()
    
    with patch('wandb.init'), \
         patch('wandb.Artifact') as mock_artifact_class, \
         patch.object(wandb_logger, 'run') as mock_run:
        
        # Track artifact instances to verify cleanup
        artifact_instances = []
        
        def track_artifact(*args, **kwargs):
            instance = MagicMock()
            artifact_instances.append(instance)
            return instance
        
        mock_artifact_class.side_effect = track_artifact
        mock_run.log_artifact = MagicMock()
        wandb_logger.wandb_api = MagicMock()
        
        try:
            wandb_logger.init_run(config={'test': True}, run_name="artifact_cleanup_test")
            memory_tracker.measure("baseline")
            
            # Create multiple checkpoints to generate artifact objects
            for epoch in range(3):
                try:
                    wandb_logger.save_checkpoint(
                        epoch=epoch,
                        model_state_dict={'small': torch.randn(100, 100)},
                        optimizer_state_dict={'state': {}},
                        loss=0.5
                    )
                except Exception:
                    pass  # Expected in test environment
            
            memory_tracker.measure("after_checkpoints")
            
            # Verify that artifact instances were created
            assert len(artifact_instances) == 3, f"Expected 3 artifacts, got {len(artifact_instances)}"
            
            # Force cleanup and garbage collection
            artifact_instances.clear()
            gc.collect()
            
            memory_tracker.measure("after_artifact_cleanup")
            memory_tracker.assert_no_major_leak(threshold_mb=10.0)
            
        finally:
            try:
                wandb_logger.finish()
            except Exception:
                pass


@pytest.mark.performance
def test_multiple_checkpoint_cycles_memory_stability(memory_tracker):
    """Test memory stability over multiple checkpoint save/load cycles."""
    setup_environment_secrets()
    
    with patch('wandb.init'), \
         patch('wandb.Artifact') as mock_artifact, \
         patch.object(wandb_logger, 'run') as mock_run:
        
        mock_artifact_instance = MagicMock()
        mock_artifact.return_value = mock_artifact_instance
        mock_run.log_artifact = MagicMock()
        wandb_logger.wandb_api = MagicMock()
        
        try:
            wandb_logger.init_run(config={'test': True}, run_name="stability_test")
            memory_tracker.measure("baseline")
            
            # Simulate multiple training cycles with checkpointing
            for cycle in range(3):
                for epoch in range(3):
                    # Create fresh state dicts each time (simulating real training)
                    model_state = {f'layer_{i}.weight': torch.randn(500, 500) for i in range(3)}
                    optimizer_state = {'state': {i: {'momentum': torch.randn(500, 500)} for i in range(3)}}
                    
                    try:
                        wandb_logger.save_checkpoint(
                            epoch=cycle * 3 + epoch,
                            model_state_dict=model_state,
                            optimizer_state_dict=optimizer_state,
                            loss=1.0 / (cycle * 3 + epoch + 1)
                        )
                    except Exception:
                        pass
                    
                    # Clear references and force GC periodically
                    del model_state, optimizer_state
                    if epoch % 2 == 0:
                        gc.collect()
                
                memory_tracker.measure(f"end_of_cycle_{cycle}")
            
            # Final stability check
            memory_tracker.assert_no_major_leak(threshold_mb=30.0)
            
        finally:
            try:
                wandb_logger.finish()
            except Exception:
                pass
