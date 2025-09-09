"""
Thread-safe utilities for handling multiprocessing and multithreading in ML training.
This module provides safe alternatives to avoid fork crashes while maintaining performance.
"""

import os
import sys
import threading
import multiprocessing as mp
from contextlib import contextmanager
from typing import Optional, Any, Dict
import torch
import logging

logger = logging.getLogger(__name__)


class ThreadSafetyManager:
    """Manages thread safety configuration for the entire application."""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_thread_safety()
            self._initialized = True

    def _setup_thread_safety(self):
        """Configure thread safety settings."""
        # Set multiprocessing start method to 'spawn' for macOS safety
        try:
            if mp.get_start_method(allow_none=True) is None:
                if sys.platform == "darwin":  # macOS
                    mp.set_start_method("spawn", force=True)
                else:
                    # Use spawn for all platforms for consistency
                    mp.set_start_method("spawn", force=True)
                logger.info(f"Set multiprocessing start method to: {mp.get_start_method()}")
        except RuntimeError as e:
            logger.warning(f"Could not set multiprocessing start method: {e}")

    def configure_for_training(
        self, num_workers: Optional[int] = None, enable_multithreading: bool = True
    ):
        """
        Configure threading settings for training.

        Args:
            num_workers: Number of DataLoader workers. If None, auto-detect optimal number.
            enable_multithreading: Whether to enable multithreading for performance.
        """
        if enable_multithreading:
            # Calculate optimal number of workers
            if num_workers is None:
                cpu_count = os.cpu_count() or 1
                # Use fewer workers on macOS to avoid fork issues
                if sys.platform == "darwin":
                    num_workers = min(4, max(1, cpu_count // 2))
                else:
                    num_workers = min(8, max(1, cpu_count - 1))

            # Ensure num_workers is always positive for multithreading
            num_workers = max(1, num_workers)

            # Set conservative threading limits that allow multithreading but prevent conflicts
            thread_count = max(1, min(4, num_workers))  # Ensure at least 1 thread
            os.environ["OMP_NUM_THREADS"] = str(thread_count)
            os.environ["MKL_NUM_THREADS"] = str(thread_count)
            os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)

            # Configure PyTorch threading (must be positive)
            torch.set_num_threads(thread_count)

            logger.info(f"Configured for multithreaded training with {num_workers} workers")

        else:
            # Conservative single-threaded mode
            num_workers = 0
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            torch.set_num_threads(1)

            logger.info("Configured for single-threaded training")

        # Set additional safety variables
        os.environ["TORCH_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")

        return num_workers

    def get_safe_dataloader_kwargs(
        self, num_workers: Optional[int] = None, enable_multithreading: bool = True
    ) -> Dict[str, Any]:
        """
        Get safe DataLoader configuration.

        Args:
            num_workers: Number of workers for DataLoader
            enable_multithreading: Whether to enable multithreading

        Returns:
            Dictionary of DataLoader kwargs
        """
        if num_workers is None:
            num_workers = self.configure_for_training(enable_multithreading=enable_multithreading)

        # Configure DataLoader for thread safety
        dataloader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        }

        # Additional safety measures for macOS
        if sys.platform == "darwin" and num_workers > 0:
            dataloader_kwargs.update(
                {
                    "multiprocessing_context": mp.get_context("spawn"),
                    "worker_init_fn": self._worker_init_fn,
                }
            )

        return {k: v for k, v in dataloader_kwargs.items() if v is not None}

    @staticmethod
    def _worker_init_fn(worker_id: int):
        """Initialize DataLoader workers safely."""
        # Set random seeds for reproducibility
        import random
        import numpy as np

        # Set worker-specific random seeds
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        # Configure worker threading
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)


@contextmanager
def safe_multiprocessing_context(
    enable_multithreading: bool = True, num_workers: Optional[int] = None
):
    """
    Context manager for safe multiprocessing operations.

    Args:
        enable_multithreading: Whether to enable multithreading
        num_workers: Number of workers to use

    Example:
        with safe_multiprocessing_context(enable_multithreading=True, num_workers=4):
            # Your training code here
            dataloader = DataLoader(dataset, **safety_manager.get_safe_dataloader_kwargs())
    """
    safety_manager = ThreadSafetyManager()

    # Store original environment variables
    original_env = {
        key: os.environ.get(key)
        for key in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TORCH_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ]
    }

    original_torch_threads = torch.get_num_threads()

    try:
        # Configure for the operation
        actual_num_workers = safety_manager.configure_for_training(
            num_workers=num_workers, enable_multithreading=enable_multithreading
        )

        yield safety_manager, actual_num_workers

    finally:
        # Restore original settings
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

        torch.set_num_threads(original_torch_threads)


def configure_torch_for_safety():
    """Configure PyTorch for fork safety and optimal performance."""
    # Disable fork in PyTorch multiprocessing
    if hasattr(torch.multiprocessing, "set_start_method"):
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # Configure CUDA settings for thread safety
    if torch.cuda.is_available():
        # Prevent CUDA context issues in forked processes
        torch.cuda.set_device(0)  # Ensure we're using the primary GPU

        # Configure CUDA memory management for stability
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_optimal_worker_count(
    enable_multithreading: bool = True, max_workers: Optional[int] = None
) -> int:
    """
    Calculate optimal number of workers for current system.

    Args:
        enable_multithreading: Whether multithreading is enabled
        max_workers: Maximum number of workers to use

    Returns:
        Optimal number of workers
    """
    if not enable_multithreading:
        return 0

    cpu_count = os.cpu_count() or 1

    if sys.platform == "darwin":  # macOS
        # Conservative approach for macOS to avoid fork issues
        optimal = min(4, max(1, cpu_count // 2))
    else:
        # More aggressive on Linux/Windows
        optimal = min(8, max(1, cpu_count - 1))

    if max_workers is not None:
        optimal = min(optimal, max_workers)

    return optimal


# Initialize the singleton when module is imported
_safety_manager = ThreadSafetyManager()

# Configure PyTorch on import
configure_torch_for_safety()
