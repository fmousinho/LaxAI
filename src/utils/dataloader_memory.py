"""
DataLoader Memory Optimization Utilities

This module provides utilities to optimize DataLoader configuration and prevent
memory leaks from worker processes and caching.
"""

import os
import torch
import logging
import gc
from torch.utils.data import DataLoader
from typing import Optional, Any, Dict
import multiprocessing as mp

logger = logging.getLogger(__name__)

def get_optimal_dataloader_config(
    num_workers: Optional[int] = None,
    batch_size: int = 32,
    dataset_size: Optional[int] = None,
    memory_constrained: bool = False
) -> Dict[str, Any]:
    """
    Get optimal DataLoader configuration for memory efficiency.

    Args:
        num_workers: Number of workers (auto-detected if None)
        batch_size: Batch size for training
        dataset_size: Size of dataset (for prefetch calculation)
        memory_constrained: If True, use more conservative settings

    Returns:
        Dictionary with optimal DataLoader kwargs
    """
    # Determine optimal number of workers
    if num_workers is None:
        if memory_constrained:
            # Conservative approach for memory-constrained environments
            num_workers = min(2, mp.cpu_count())
        else:
            # Use up to 75% of available CPUs, max 8
            num_workers = min(max(1, int(mp.cpu_count() * 0.75)), 8)

    # Calculate prefetch factor based on dataset size and memory constraints
    prefetch_factor = None
    if num_workers > 0:
        if memory_constrained:
            prefetch_factor = 1  # Minimal prefetching
        elif dataset_size and dataset_size > 10000:
            prefetch_factor = 4  # Higher prefetch for large datasets
        else:
            prefetch_factor = 2  # Default prefetch

    config = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available() and num_workers > 0,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': prefetch_factor,
        'drop_last': True  # Consistent batch sizes
    }

    logger.info(f"Optimal DataLoader config: workers={num_workers}, prefetch={prefetch_factor}, pin_memory={config['pin_memory']}")
    return config

def cleanup_dataloader_workers(dataloader: DataLoader):
    """
    Properly cleanup DataLoader worker processes to prevent memory leaks.

    Args:
        dataloader: The DataLoader to cleanup
    """
    try:
        # Shutdown the worker processes
        if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
            if hasattr(dataloader._iterator, '_shutdown_workers'):
                dataloader._iterator._shutdown_workers()

        # Force garbage collection
        gc.collect()

        logger.debug("DataLoader workers cleaned up successfully")
    except Exception as e:
        logger.warning(f"Failed to cleanup DataLoader workers: {e}")

def monitor_dataloader_memory(dataloader: DataLoader, context: str = ""):
    """
    Monitor memory usage of DataLoader and its workers.

    Args:
        dataloader: DataLoader to monitor
        context: Context description for logging
    """
    try:
        import psutil
        process = psutil.Process()

        # Get worker process information if available
        worker_info = ""
        if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
            iterator = dataloader._iterator
            if hasattr(iterator, '_workers') and iterator._workers:
                worker_count = len(iterator._workers)
                worker_info = f", {worker_count} workers"

        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"{context} - DataLoader Memory: {memory_mb:.1f}MB{worker_info}")

    except ImportError:
        logger.debug(f"{context} - DataLoader monitoring unavailable (psutil not installed)")
    except Exception as e:
        logger.warning(f"Failed to monitor DataLoader memory: {e}")

class MemoryEfficientDataLoader:
    """
    A wrapper around PyTorch DataLoader with memory optimization features.
    """

    def __init__(self, dataset, batch_size: int = 32, memory_constrained: bool = False, **kwargs):
        """
        Initialize memory-efficient DataLoader.

        Args:
            dataset: The dataset to load
            batch_size: Batch size
            memory_constrained: Use conservative memory settings
            **kwargs: Additional DataLoader arguments
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Get optimal configuration
        optimal_config = get_optimal_dataloader_config(
            num_workers=kwargs.get('num_workers'),
            batch_size=batch_size,
            dataset_size=len(dataset) if hasattr(dataset, '__len__') else None,
            memory_constrained=memory_constrained
        )

        # Merge with user-provided kwargs (user kwargs take precedence)
        dataloader_kwargs = {**optimal_config, **kwargs}
        dataloader_kwargs['batch_size'] = batch_size

        self.dataloader = DataLoader(dataset, **dataloader_kwargs)
        self.memory_constrained = memory_constrained

        logger.info(f"MemoryEfficientDataLoader initialized with {dataloader_kwargs}")

    def __iter__(self):
        """Iterate over the DataLoader."""
        return iter(self.dataloader)

    def __len__(self):
        """Return length of DataLoader."""
        return len(self.dataloader)

    @property
    def num_workers(self):
        """Get number of workers."""
        return self.dataloader.num_workers

    @property
    def pin_memory(self):
        """Get pin_memory setting."""
        return getattr(self.dataloader, 'pin_memory', False)

    @property
    def persistent_workers(self):
        """Get persistent_workers setting."""
        return getattr(self.dataloader, 'persistent_workers', False)

    def cleanup(self):
        """Clean up DataLoader resources."""
        cleanup_dataloader_workers(self.dataloader)

    def monitor_memory(self, context: str = ""):
        """Monitor DataLoader memory usage."""
        monitor_dataloader_memory(self.dataloader, context)

def create_memory_efficient_dataloader(
    dataset,
    batch_size: int = 32,
    memory_constrained: bool = False,
    **kwargs
) -> MemoryEfficientDataLoader:
    """
    Factory function to create a memory-efficient DataLoader.

    Args:
        dataset: The dataset to load
        batch_size: Batch size
        memory_constrained: Use conservative memory settings
        **kwargs: Additional DataLoader arguments

    Returns:
        MemoryEfficientDataLoader instance
    """
    return MemoryEfficientDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        memory_constrained=memory_constrained,
        **kwargs
    )
