"""
Performance tests for service_training.

These tests focus on performance, memory usage, and scalability:
- Memory leak detection and monitoring
- Performance benchmarking
- Resource usage analysis
- Long-running tests that may take significant time
- May require special hardware (GPU) or large datasets
"""

import os
import sys
from pathlib import Path

import psutil
import pytest
import torch

# Add src and shared_libs to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'service_training' / 'src'))
sys.path.insert(0, str(project_root / 'shared_libs'))

@pytest.fixture
def performance_config():
    """Configuration optimized for performance testing."""
    return {
        'batch_size': 16,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'memory_monitoring': True,
        'performance_tracking': True
    }

@pytest.fixture
def memory_monitor():
    """Memory monitoring fixture for performance tests."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    class MemoryMonitor:
        def __init__(self, initial_mb):
            self.initial_mb = initial_mb
            self.peak_mb = initial_mb

        def check_memory(self):
            current = process.memory_info().rss / 1024 / 1024
            self.peak_mb = max(self.peak_mb, current)
            return current

        def get_memory_stats(self):
            return {
                'initial_mb': self.initial_mb,
                'current_mb': self.check_memory(),
                'peak_mb': self.peak_mb,
                'increase_mb': self.peak_mb - self.initial_mb
            }

    return MemoryMonitor(initial_memory)

@pytest.fixture
def gpu_available():
    """Check if GPU is available for performance testing."""
    return torch.cuda.is_available()

@pytest.mark.performance
def pytest_configure(config):
    """Configure pytest for performance tests."""
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "memory: mark test as a memory-related performance test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
