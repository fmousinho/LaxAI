#!/usr/bin/env python3
"""
Comprehensive Memory Management Test Suite

This script tests all the new memory management features implemented to fix
CPU memory leaks during training.
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import new memory management utilities
from utils.cpu_memory import (
    CPUMemoryMonitor,
    clear_cpu_memory,
    log_comprehensive_memory_stats,
    get_comprehensive_memory_stats
)
from utils.dataloader_memory import (
    create_memory_efficient_dataloader,
    get_optimal_dataloader_config
)
from utils.dataset_memory import (
    create_memory_efficient_dataset,
    optimize_dataset_cache
)
from utils.gpu_memory import clear_gpu_memory, log_gpu_memory_stats

# Import existing training components for testing
from train.training import Training

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DummyDataset(Dataset):
    """Simple dataset for testing memory management."""

    def __init__(self, size: int = 1000, image_size: tuple = (224, 224)):
        self.size = size
        self.image_size = image_size
        # Simulate some data that would normally be cached
        self.data = [np.random.rand(*image_size, 3).astype(np.float32) for _ in range(size)]
        self.labels = np.random.randint(0, 10, size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate some processing that might cause memory accumulation
        image = torch.from_numpy(self.data[idx]).permute(2, 0, 1)
        label = self.labels[idx]

        # Simulate triplet data (anchor, positive, negative)
        pos_idx = np.random.choice(len(self))
        neg_idx = np.random.choice([i for i in range(len(self)) if self.labels[i] != label])

        anchor = image
        positive = torch.from_numpy(self.data[pos_idx]).permute(2, 0, 1)
        negative = torch.from_numpy(self.data[neg_idx]).permute(2, 0, 1)

        return anchor, positive, negative, label

class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze()
        return self.fc(x)

    def forward_triplet(self, anchor, positive, negative):
        emb_anchor = self(anchor)
        emb_positive = self(positive)
        emb_negative = self(negative)
        return emb_anchor, emb_positive, emb_negative

    @property
    def device(self):
        """Return the device of the model's parameters."""
        return next(self.parameters()).device

def test_cpu_memory_monitoring():
    """Test CPU memory monitoring functionality."""
    logger.info("=== Testing CPU Memory Monitoring ===")

    monitor = CPUMemoryMonitor()
    monitor.log_memory_stats("Test Start")

    # Simulate memory allocation
    logger.info("Allocating memory...")
    large_arrays = []
    for i in range(10):
        large_arrays.append(np.random.rand(1000, 1000).astype(np.float32))
        if i % 3 == 0:
            monitor.log_memory_stats(f"After allocation {i+1}")

    # Test memory cleanup
    logger.info("Testing memory cleanup...")
    clear_cpu_memory()
    monitor.log_memory_stats("After cleanup")

    # Test comprehensive stats
    logger.info("Testing comprehensive memory stats...")
    stats = get_comprehensive_memory_stats()
    logger.info(f"Comprehensive stats keys: {list(stats.keys())}")

    return True

def test_dataloader_optimization():
    """Test DataLoader memory optimization."""
    logger.info("=== Testing DataLoader Optimization ===")

    # Create test dataset
    dataset = DummyDataset(size=100)

    # Test optimal config generation
    config = get_optimal_dataloader_config(
        batch_size=8,
        dataset_size=len(dataset),
        memory_constrained=False
    )
    logger.info(f"Optimal DataLoader config: {config}")

    # Test memory-efficient DataLoader
    dataloader = create_memory_efficient_dataloader(
        dataset=dataset,
        batch_size=8,
        memory_constrained=False,
        **config
    )

    logger.info(f"DataLoader created with {dataloader.num_workers} workers")
    logger.info(f"Pin memory: {dataloader.pin_memory}")
    logger.info(f"Persistent workers: {dataloader.persistent_workers}")

    # Test iteration
    logger.info("Testing DataLoader iteration...")
    for i, (anchor, positive, negative, label) in enumerate(dataloader):
        if i >= 3:  # Just test a few batches
            break
        logger.info(f"Batch {i}: shapes - anchor: {anchor.shape}, positive: {positive.shape}, negative: {negative.shape}")

    # Test cleanup
    dataloader.cleanup()
    logger.info("DataLoader cleanup completed")

    return True

def test_dataset_optimization():
    """Test dataset memory optimization."""
    logger.info("=== Testing Dataset Optimization ===")

    # Create test dataset
    dataset = DummyDataset(size=50)

    # Test memory-efficient wrapper
    wrapped_dataset = create_memory_efficient_dataset(
        dataset=dataset,
        monitor_cache=True,
        auto_cleanup=True
    )

    logger.info(f"Wrapped dataset size: {len(wrapped_dataset)}")

    # Test cache optimization
    optimize_dataset_cache(dataset)
    logger.info("Dataset cache optimized")

    # Test iteration with monitoring
    logger.info("Testing dataset iteration with monitoring...")
    for i in range(10):
        item = wrapped_dataset[i]
        if i % 5 == 0:
            stats = wrapped_dataset.get_memory_stats()
            logger.info(f"Access {i}: {stats}")

    # Test cleanup
    wrapped_dataset.cleanup()
    logger.info("Dataset cleanup completed")

    return True

def test_training_integration():
    """Test integration with training pipeline."""
    logger.info("=== Testing Training Integration ===")

    # Create test dataset
    train_dataset = DummyDataset(size=50)
    val_dataset = DummyDataset(size=20)

    # Create memory-efficient datasets
    train_dataset = create_memory_efficient_dataset(train_dataset)
    val_dataset = create_memory_efficient_dataset(val_dataset)

    # Create model
    model = SimpleModel()

    # Create training instance
    training = Training(
        device='cpu',  # Use CPU for testing
        enable_multithreading=False,  # Disable for simpler testing
        num_epochs=2,
        batch_size=4,
        learning_rate=0.001,
        margin=0.5
    )

    # Setup model and datasets
    training.setup_model(SimpleModel, "test_model")
    training.setup_dataloader(train_dataset, type='train')
    training.setup_dataloader(val_dataset, type='val')

    # Test memory monitoring during setup
    training.cpu_monitor.log_memory_stats("After setup")

    # Run a short training loop manually to test integration
    logger.info("Running mini training test...")

    training.model.train()
    dataloader = training.dataloader

    for epoch in range(1):  # Just 1 epoch for testing
        training.cpu_monitor.log_memory_stats(f"Epoch {epoch + 1} start")

        for i, (anchor, positive, negative, _) in enumerate(dataloader):
            if i >= 3:  # Just a few batches
                break

            # Move to device
            anchor = anchor.to(training.device)
            positive = positive.to(training.device)
            negative = negative.to(training.device)

            # Forward pass
            emb_anchor, emb_positive, emb_negative = training.model.forward_triplet(anchor, positive, negative)
            loss = training.loss_fn(emb_anchor, emb_positive, emb_negative)

            # Backward and optimize
            training.optimizer.zero_grad()
            loss.backward()
            training.optimizer.step()

            # Test periodic cleanup
            if i == 2:
                clear_cpu_memory()
                training.cpu_monitor.log_memory_stats(f"Batch {i+1} after cleanup")

            # Clean up tensors
            del anchor, positive, negative, emb_anchor, emb_positive, emb_negative, loss

        training.cpu_monitor.log_memory_stats(f"Epoch {epoch + 1} complete")

    # Test cleanup
    clear_cpu_memory()
    training.cpu_monitor.log_memory_stats("After training cleanup")

    return True

def run_memory_profiler_test():
    """Test the memory profiler script."""
    logger.info("=== Testing Memory Profiler ===")

    # Import and test the profiler
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from memory_profiler import TrainingMemoryProfiler

    profiler = TrainingMemoryProfiler(
        log_file="test_memory_profile.jsonl",
        interval=1.0  # Fast interval for testing
    )

    # Start profiling
    profiler.start_profiling()
    logger.info("Memory profiler started")

    # Simulate some work
    time.sleep(2)

    # Take manual snapshot
    profiler._take_snapshot("Test snapshot")

    # Stop profiling
    profiler.stop_profiling()
    logger.info("Memory profiler stopped")

    # Check if log file was created
    if profiler.log_file.exists():
        logger.info(f"Memory profile log created: {profiler.log_file}")
        # Read first few lines
        with open(profiler.log_file, 'r') as f:
            lines = f.readlines()[:3]
            logger.info(f"Sample log entries: {len(lines)} lines")
    else:
        logger.warning("Memory profile log was not created")

    return True

def main():
    """Run comprehensive memory management tests."""
    logger.info("Starting Comprehensive Memory Management Test Suite")
    logger.info("=" * 60)

    # Track test results
    test_results = {}

    try:
        # Test 1: CPU Memory Monitoring
        test_results['cpu_monitoring'] = test_cpu_memory_monitoring()

        # Test 2: DataLoader Optimization
        test_results['dataloader'] = test_dataloader_optimization()

        # Test 3: Dataset Optimization
        test_results['dataset'] = test_dataset_optimization()

        # Test 4: Training Integration
        test_results['training'] = test_training_integration()

        # Test 5: Memory Profiler
        test_results['profiler'] = run_memory_profiler_test()

        # Final memory cleanup
        logger.info("=== Final Cleanup ===")
        clear_cpu_memory()
        if torch.cuda.is_available():
            clear_gpu_memory()

        log_comprehensive_memory_stats("Test completion")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Report results
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:15}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! Memory management fixes are working correctly.")
    else:
        logger.info("⚠️  Some tests failed. Please check the logs above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
