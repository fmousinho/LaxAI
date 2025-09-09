"""
CPU Memory Management Utilities for Training

This module provides comprehensive CPU memory monitoring and cleanup utilities
to prevent memory leaks during training epochs.
"""

import gc
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Check if psutil is available
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. CPU memory monitoring will be limited.")


class CPUMemoryMonitor:
    """Monitor CPU memory usage throughout training."""

    def __init__(self):
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.baseline_memory = self._get_current_memory()

    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if not PSUTIL_AVAILABLE or self.process is None:
            return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}

        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": memory_percent,
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current = self._get_current_memory()
        baseline = self.baseline_memory

        return {
            "current": current,
            "baseline": baseline,
            "increase_mb": current["rss_mb"] - baseline["rss_mb"],
            "increase_percent": (
                ((current["rss_mb"] - baseline["rss_mb"]) / baseline["rss_mb"]) * 100
                if baseline["rss_mb"] > 0
                else 0
            ),
        }

    def log_memory_stats(self, context: str = ""):
        """Log current memory statistics."""
        if not PSUTIL_AVAILABLE:
            logger.info(f"{context} - CPU memory monitoring unavailable (psutil not installed)")
            return

        stats = self.get_memory_stats()
        current = stats["current"]

        log_msg = f"{context} - CPU Memory: {current['rss_mb']:.1f}MB RSS, {current['vms_mb']:.1f}MB VMS ({current['percent']:.1f}%)"

        if stats["increase_mb"] > 0:
            log_msg += (
                f" | Increase: +{stats['increase_mb']:.1f}MB ({stats['increase_percent']:.1f}%)"
            )

        logger.info(log_msg)


def clear_cpu_memory(force: bool = False):
    """
    Comprehensive CPU memory cleanup.

    Args:
        force: If True, use more aggressive cleanup methods
    """
    # Force garbage collection
    collected = gc.collect()
    if collected > 0:
        logger.debug(f"Garbage collector freed {collected} objects")

    # Clear any cached objects
    if hasattr(gc, "garbage") and gc.garbage:
        logger.warning(f"Found {len(gc.garbage)} uncollectable objects")
        if force:
            gc.garbage.clear()

    # Additional cleanup for numpy
    try:
        # This helps with numpy array cleanup by forcing garbage collection on numpy objects
        import numpy as np

        # Force cleanup of numpy's internal memory pools
        # Note: This is a more conservative approach than calling private methods
        pass  # Numpy cleanup is handled by Python's GC
    except ImportError:
        pass

    # Force cleanup of any remaining references
    if force:
        # Clear any module-level caches that might exist
        import sys

        for module_name, module in sys.modules.items():
            if hasattr(module, "_cache") and isinstance(module._cache, dict):
                module._cache.clear()


@contextmanager
def cpu_memory_context(monitor: Optional[CPUMemoryMonitor] = None, context: str = ""):
    """
    Context manager for monitoring CPU memory usage around a code block.

    Args:
        monitor: CPUMemoryMonitor instance to use
        context: Description of the code block being monitored
    """
    if monitor:
        monitor.log_memory_stats(f"{context} - Start")

    try:
        yield
    finally:
        if monitor:
            monitor.log_memory_stats(f"{context} - End")
        clear_cpu_memory()


def get_comprehensive_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory statistics including system info."""
    stats = {}

    if PSUTIL_AVAILABLE:
        process = psutil.Process()

        # Process memory
        memory_info = process.memory_info()
        stats.update(
            {
                "process_rss_mb": memory_info.rss / 1024 / 1024,
                "process_vms_mb": memory_info.vms / 1024 / 1024,
                "process_percent": process.memory_percent(),
            }
        )

        # System memory
        system_memory = psutil.virtual_memory()
        stats.update(
            {
                "system_total_mb": system_memory.total / 1024 / 1024,
                "system_available_mb": system_memory.available / 1024 / 1024,
                "system_percent": system_memory.percent,
            }
        )

    # Python garbage collector stats
    gc_stats = {
        "gc_objects": len(gc.get_objects()),
        "gc_collections": {i: gc.get_count()[i] for i in range(3)},
        "gc_garbage_count": len(gc.garbage),
    }
    stats["gc"] = gc_stats

    # GPU memory if available
    if torch.cuda.is_available():
        stats.update(
            {
                "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
        )

    return stats


def log_comprehensive_memory_stats(context: str = ""):
    """Log comprehensive memory statistics."""
    stats = get_comprehensive_memory_stats()

    log_msg = f"{context} - Memory Stats:\n"

    if "process_rss_mb" in stats:
        log_msg += (
            f"  Process: {stats['process_rss_mb']:.1f}MB RSS ({stats['process_percent']:.1f}%)\n"
        )
        log_msg += f"  System: {stats['system_available_mb']:.1f}MB available of {stats['system_total_mb']:.1f}MB ({stats['system_percent']:.1f}%)\n"

    if "gpu_allocated_mb" in stats:
        log_msg += f"  GPU: {stats['gpu_allocated_mb']:.1f}MB allocated, {stats['gpu_reserved_mb']:.1f}MB reserved\n"

    log_msg += (
        f"  GC: {stats['gc']['gc_objects']} objects, {stats['gc']['gc_garbage_count']} garbage"
    )

    logger.info(log_msg)
