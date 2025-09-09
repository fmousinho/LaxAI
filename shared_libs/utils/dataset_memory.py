"""
Dataset Memory Optimization Utilities

This module provides utilities to optimize dataset memory usage and prevent
memory leaks from caching mechanisms.
"""

import gc
import logging
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DatasetMemoryMonitor:
    """Monitor dataset memory usage and cache statistics."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.initial_cache_size = self._get_cache_size()

    def _get_cache_size(self) -> int:
        """Get current cache size if dataset has caching."""
        if hasattr(self.dataset, "_image_cache"):
            cache = getattr(self.dataset, "_image_cache")
            if hasattr(cache, "__len__"):
                return len(cache)
        return 0

    def _get_cache_memory_usage(self) -> float:
        """Estimate memory usage of cache in MB."""
        try:
            cache_size = self._get_cache_size()
            if cache_size == 0:
                return 0.0

            # Estimate ~10MB per cached image (rough estimate)
            # This can be made more accurate by inspecting actual cache contents
            estimated_mb = cache_size * 10.0
            return estimated_mb
        except Exception:
            return 0.0

    def log_cache_stats(self, context: str = ""):
        """Log cache statistics and memory usage."""
        cache_size = self._get_cache_size()
        cache_memory = self._get_cache_memory_usage()

        if cache_size > 0:
            logger.info(f"{context} - Dataset Cache: {cache_size} items, ~{cache_memory:.1f}MB")
        else:
            logger.debug(f"{context} - Dataset Cache: No cache or empty")


def optimize_dataset_cache(dataset: Dataset, max_cache_size: Optional[int] = None):
    """
    Optimize dataset cache settings for memory efficiency.

    Args:
        dataset: The dataset to optimize
        max_cache_size: Maximum cache size (if None, uses default optimization)
    """
    if not hasattr(dataset, "_image_cache"):
        logger.debug("Dataset does not have image cache to optimize")
        return

    cache = getattr(dataset, "_image_cache")

    # Set optimal cache size based on available memory
    if max_cache_size is None:
        try:
            import psutil

            memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
            # Use 10% of available memory for cache, assuming ~10MB per image
            max_cache_size = int((memory_gb * 0.1) * 100)
        except ImportError:
            # Fallback: conservative cache size
            max_cache_size = 100

    # Apply cache size limit if cache supports it
    if hasattr(cache, "maxsize"):
        original_maxsize = getattr(cache, "maxsize", None)
        cache.maxsize = max_cache_size
        logger.info(f"Dataset cache maxsize set to {max_cache_size} (was {original_maxsize})")


def clear_dataset_cache(dataset: Dataset, force: bool = False):
    """
    Clear dataset cache to free memory.

    Args:
        dataset: The dataset whose cache to clear
        force: If True, use more aggressive clearing
    """
    try:
        if hasattr(dataset, "_image_cache"):
            cache = getattr(dataset, "_image_cache")

            # Clear cache if it has a clear method
            if hasattr(cache, "clear"):
                cache.clear()
                logger.debug("Dataset image cache cleared")

            # For LRU cache, also clear internal structures
            if hasattr(cache, "cache_info"):
                info = cache.cache_info()
                logger.debug(f"Cache cleared - was {info}")

        # Clear any other cached data
        if hasattr(dataset, "_cached_data") and force:
            dataset._cached_data.clear()  # type: ignore

        # Force garbage collection
        gc.collect()

    except Exception as e:
        logger.warning(f"Failed to clear dataset cache: {e}")


def monitor_dataset_memory(dataset: Dataset, context: str = ""):
    """
    Monitor dataset memory usage.

    Args:
        dataset: Dataset to monitor
        context: Context description for logging
    """
    monitor = DatasetMemoryMonitor(dataset)
    monitor.log_cache_stats(context)

    # Additional memory monitoring
    try:
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"{context} - Process Memory: {memory_mb:.1f}MB")
    except ImportError:
        pass


class MemoryEfficientDatasetWrapper:
    """
    Wrapper around dataset to add memory monitoring and optimization.
    """

    def __init__(self, dataset: Dataset, monitor_cache: bool = True, auto_cleanup: bool = True):
        """
        Initialize memory-efficient dataset wrapper.

        Args:
            dataset: The original dataset
            monitor_cache: Whether to monitor cache usage
            auto_cleanup: Whether to automatically cleanup cache periodically
        """
        self.dataset = dataset
        self.monitor_cache = monitor_cache
        self.auto_cleanup = auto_cleanup
        self.monitor = DatasetMemoryMonitor(dataset) if monitor_cache else None
        self.access_count = 0

        # Optimize cache settings
        optimize_dataset_cache(dataset)

        logger.info("MemoryEfficientDatasetWrapper initialized")

    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        """Get item from dataset with memory monitoring."""
        self.access_count += 1

        # Periodic cleanup
        if self.auto_cleanup and self.access_count % 1000 == 0:
            clear_dataset_cache(self.dataset)
            if self.monitor:
                self.monitor.log_cache_stats(f"Access {self.access_count}")

        return self.dataset[idx]

    def cleanup(self):
        """Clean up dataset resources."""
        clear_dataset_cache(self.dataset, force=True)
        logger.info("Dataset wrapper cleanup completed")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "access_count": self.access_count,
            "cache_size": self.monitor._get_cache_size() if self.monitor else 0,
            "cache_memory_mb": self.monitor._get_cache_memory_usage() if self.monitor else 0.0,
        }
        return stats


def create_memory_efficient_dataset(
    dataset: Dataset, monitor_cache: bool = True, auto_cleanup: bool = True
) -> MemoryEfficientDatasetWrapper:
    """
    Factory function to create memory-efficient dataset wrapper.

    Args:
        dataset: Original dataset
        monitor_cache: Whether to monitor cache usage
        auto_cleanup: Whether to automatically cleanup cache

    Returns:
        MemoryEfficientDatasetWrapper instance
    """
    return MemoryEfficientDatasetWrapper(
        dataset=dataset, monitor_cache=monitor_cache, auto_cleanup=auto_cleanup
    )
