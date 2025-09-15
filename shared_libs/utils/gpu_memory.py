"""
GPU Memory Management Utilities

This module provides utilities for managing GPU memory in PyTorch,
including cleanup functions for handling OOM errors and memory leaks.
"""

import torch
import gc
import logging

logger = logging.getLogger(__name__)


def clear_gpu_memory(force: bool = False) -> None:
    """
    Comprehensive GPU memory cleanup function.

    This function performs a thorough cleanup of GPU memory by:
    1. Clearing PyTorch's CUDA cache
    2. Running garbage collection
    3. Optionally forcing GPU memory reset

    Args:
        force: If True, performs more aggressive cleanup operations

    Note:
        This function should be called when recovering from OOM errors
        or at the start of training to ensure clean GPU state.
    """
    if not torch.cuda.is_available():
        logger.debug("CUDA not available, skipping GPU memory cleanup")
        return

    try:
        # Get initial memory stats
        initial_allocated = torch.cuda.memory_allocated()
        initial_cached = torch.cuda.memory_reserved()

        logger.info(
            f"GPU memory before cleanup - Allocated: {initial_allocated / 1024**3:.2f}GB, Cached: {initial_cached / 1024**3:.2f}GB"
        )

        # Step 1: Clear PyTorch's CUDA cache
        torch.cuda.empty_cache()

        # Step 2: Force garbage collection
        gc.collect()

        # Step 3: Additional cleanup for force mode
        if force:
            # Clear all CUDA tensors that might be lingering
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        del obj
                except Exception:
                    pass

            # Run garbage collection again after tensor cleanup
            gc.collect()
            torch.cuda.empty_cache()

        # Get final memory stats
        final_allocated = torch.cuda.memory_allocated()
        final_cached = torch.cuda.memory_reserved()

        freed_allocated = initial_allocated - final_allocated
        freed_cached = initial_cached - final_cached

        logger.info(
            f"GPU memory after cleanup - Allocated: {final_allocated / 1024**3:.2f}GB, Cached: {final_cached / 1024**3:.2f}GB"
        )
        logger.info(
            f"Memory freed - Allocated: {freed_allocated / 1024**3:.2f}GB, Cached: {freed_cached / 1024**3:.2f}GB"
        )

    except Exception as e:
        logger.warning(f"Error during GPU memory cleanup: {e}")


def get_gpu_memory_stats() -> dict:
    """
    Get current GPU memory usage statistics.

    Returns:
        Dictionary containing memory statistics in bytes and GB
    """
    if not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "allocated_gb": 0.0,
            "cached_bytes": 0,
            "cached_gb": 0.0,
            "max_allocated_bytes": 0,
            "max_allocated_gb": 0.0,
            "device_count": 0,
        }

    allocated_bytes = torch.cuda.memory_allocated()
    cached_bytes = torch.cuda.memory_reserved()
    max_allocated_bytes = torch.cuda.max_memory_allocated()

    return {
        "allocated_bytes": allocated_bytes,
        "allocated_gb": allocated_bytes / 1024**3,
        "cached_bytes": cached_bytes,
        "cached_gb": cached_bytes / 1024**3,
        "max_allocated_bytes": max_allocated_bytes,
        "max_allocated_gb": max_allocated_bytes / 1024**3,
        "device_count": torch.cuda.device_count(),
    }


def log_gpu_memory_stats(prefix: str = "GPU Memory") -> None:
    """
    Log current GPU memory statistics.

    Args:
        prefix: Prefix string for the log message
    """
    stats = get_gpu_memory_stats()
    if stats["device_count"] > 0:
        logger.info(
            f"{prefix} - Allocated: {stats['allocated_gb']:.2f}GB, Cached: {stats['cached_gb']:.2f}GB, Max: {stats['max_allocated_gb']:.2f}GB"
        )
    else:
        logger.info(f"{prefix} - CUDA not available")


def reset_peak_memory_stats() -> None:
    """
    Reset the peak memory statistics tracked by PyTorch.

    This is useful for tracking memory usage for specific operations
    or training phases.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.debug("GPU peak memory statistics reset")


class GPUMemoryContext:
    """
    Context manager for GPU memory monitoring and cleanup.

    Usage:
        with GPUMemoryContext("Training batch"):
            # Your GPU operations here
            pass

    This will log memory usage before and after the operations,
    and perform cleanup if an exception occurs.
    """

    def __init__(self, operation_name: str = "GPU Operation", cleanup_on_error: bool = True):
        self.operation_name = operation_name
        self.cleanup_on_error = cleanup_on_error
        self.initial_stats = None

    def __enter__(self):
        self.initial_stats = get_gpu_memory_stats()
        log_gpu_memory_stats(f"{self.operation_name} - Before")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.cleanup_on_error:
            logger.warning(
                f"Exception occurred during {self.operation_name}, performing GPU memory cleanup"
            )
            clear_gpu_memory(force=True)

        log_gpu_memory_stats(f"{self.operation_name} - After")

        if self.initial_stats:
            final_stats = get_gpu_memory_stats()
            memory_diff = final_stats["allocated_gb"] - self.initial_stats["allocated_gb"]
            if memory_diff > 0.1:  # More than 100MB difference
                logger.info(f"{self.operation_name} - Memory increase: {memory_diff:.2f}GB")
