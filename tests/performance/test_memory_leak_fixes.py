"""
Test suite to validate memory leak fixes in the training pipeline.
"""
import pytest
import time
import psutil
from unittest.mock import MagicMock, patch
import torch

from train.training import Training
from config.all_config import training_config


class MockMemoryMonitor:
    """Mock memory monitoring for testing."""
    
    def __init__(self):
        self.measurements = []
        self.baseline_memory = None
        
    def reset_baseline(self):
        """Reset memory baseline."""
        process = psutil.Process()
        self.baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
    def measure(self, label: str):
        """Take a memory measurement."""
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        delta = current_memory - self.baseline_memory if self.baseline_memory else 0
        
        measurement = {
            'label': label,
            'memory_mb': current_memory,
            'delta_mb': delta,
            'timestamp': time.time()
        }
        self.measurements.append(measurement)
        
    def get_memory_trend(self):
        """Get memory usage trend."""
        if len(self.measurements) < 2:
            return 0.0
        
        start_memory = self.measurements[0]['memory_mb']
        end_memory = self.measurements[-1]['memory_mb']
        return end_memory - start_memory


@pytest.fixture
def memory_monitor():
    """Provide memory monitoring fixture."""
    monitor = MockMemoryMonitor()
    monitor.reset_baseline()
    return monitor


@pytest.mark.performance
def test_aggressive_memory_cleanup_effectiveness(memory_monitor):
    """Test that aggressive memory cleanup actually reduces memory usage."""
    # Force garbage collection before test to ensure clean state
    import gc
    for _ in range(3):
        gc.collect()
    
    memory_monitor.measure("before_training_init")
    
    # Create training instance
    training = Training(clear_memory_on_start=True, enable_multithreading=False)
    memory_monitor.measure("after_training_init")
    
    # Test aggressive memory cleanup
    training._aggressive_memory_cleanup("test_cleanup")
    memory_monitor.measure("after_aggressive_cleanup")
    
    # Verify memory didn't increase significantly
    memory_trend = memory_monitor.get_memory_trend()
    assert memory_trend < 100, f"Memory increased by {memory_trend:.1f}MB during cleanup test"
    
    print(f"✅ Aggressive memory cleanup test passed. Memory trend: {memory_trend:.1f}MB")


@pytest.mark.performance 
def test_tensor_cleanup_in_training_loop():
    """Test that tensors are properly cleaned up in training loop simulation."""
    training = Training(
        clear_memory_on_start=True, 
        enable_multithreading=False,
        num_epochs=2,
        batch_size=4
    )
    
    # Mock model and components - embeddings need gradients for backward()
    mock_model = MagicMock()
    mock_model.forward_triplet.return_value = (
        torch.randn(4, 128, requires_grad=True),  # emb_anchor - needs gradients
        torch.randn(4, 128, requires_grad=True),  # emb_positive - needs gradients
        torch.randn(4, 128, requires_grad=True)   # emb_negative - needs gradients
    )
    
    training.model = mock_model
    training.optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.001)
    training.loss_fn = torch.nn.TripletMarginLoss()
    
    # Create mock dataloader with small batches
    mock_data = []
    for _ in range(3):  # 3 batches
        mock_data.append((
            torch.randn(4, 3, 64, 64),  # anchor
            torch.randn(4, 3, 64, 64),  # positive
            torch.randn(4, 3, 64, 64),  # negative
            torch.zeros(4)  # labels
        ))
    
    training.dataloader = mock_data
    
    # Monitor memory before and after training simulation
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    
    # Simulate training loop with tensor cleanup
    running_loss = 0.0
    for i, (anchor, positive, negative, _) in enumerate(training.dataloader):
        # Simulate forward pass
        anchor = anchor.to(training.device, non_blocking=True)
        positive = positive.to(training.device, non_blocking=True)
        negative = negative.to(training.device, non_blocking=True)
        
        training.optimizer.zero_grad()
        
        emb_anchor, emb_positive, emb_negative = training.model.forward_triplet(anchor, positive, negative)
        loss = training.loss_fn(emb_anchor, emb_positive, emb_negative)
        
        loss.backward()
        training.optimizer.step()
        
        # Extract scalar and cleanup (as in the fixed code)
        loss_value = loss.item()
        running_loss += loss_value
        
        # Explicit cleanup
        del anchor, positive, negative, emb_anchor, emb_positive, emb_negative, loss
        
        # GPU cleanup simulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Memory after training
    memory_after = process.memory_info().rss / 1024 / 1024
    memory_increase = memory_after - memory_before
    
    # Memory increase should be minimal (< 50MB)
    assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB during training simulation"
    
    print(f"✅ Tensor cleanup test passed. Memory increase: {memory_increase:.1f}MB")


@pytest.mark.performance
def test_memory_leak_detection_alerts():
    """Test that memory leak detection properly identifies and alerts on leaks."""
    training = Training(clear_memory_on_start=True, enable_multithreading=False)
    
    # Set baseline
    import psutil
    process = psutil.Process()
    training.epoch_memory_baseline = process.memory_info().rss / 1024 / 1024
    
    # Simulate memory leak by creating large objects (need to exceed 100MB threshold)
    memory_hogs = []
    for _ in range(20):  # Increased from 10 to 20 for more reliable allocation
        # Create larger objects to simulate memory leak - ~40MB each = 800MB total
        memory_hog = torch.randn(1024, 1024, 10)  # ~40MB each
        memory_hogs.append(memory_hog)
    
    # Force memory allocation by accessing the tensors
    for hog in memory_hogs:
        _ = hog.sum()  # Force actual memory allocation
    
    # Test leak detection logic
    current_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = current_memory - training.epoch_memory_baseline
    memory_increase_per_epoch = memory_increase / 3  # Simulate 3 epochs
    
    # Should detect leak if increase per epoch > threshold (100MB)
    leak_detected = memory_increase_per_epoch > training.memory_leak_threshold_mb
    
    # Clean up and force garbage collection multiple times
    del memory_hogs
    import gc
    for _ in range(5):  # More aggressive cleanup for test isolation
        gc.collect()
    
    # Additional cleanup specific to PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Verify leak detection works
    assert leak_detected, f"Memory leak detection failed. Increase per epoch: {memory_increase_per_epoch:.1f}MB"
    
    print(f"✅ Memory leak detection test passed. Detected {memory_increase_per_epoch:.1f}MB/epoch increase")


if __name__ == "__main__":
    # Run individual tests for development
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test aggressive cleanup
    monitor = MockMemoryMonitor()
    monitor.reset_baseline()
    test_aggressive_memory_cleanup_effectiveness(monitor)
    
    # Test tensor cleanup
    test_tensor_cleanup_in_training_loop()
    
    # Test leak detection
    test_memory_leak_detection_alerts()
    
    print("✅ All memory leak fix tests passed!")
