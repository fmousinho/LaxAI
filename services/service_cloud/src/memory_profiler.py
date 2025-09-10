#!/usr/bin/env python3
"""
Memory Profiling Script for Training

This script provides comprehensive memory profiling capabilities to monitor
and diagnose memory leaks during training epochs.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shared_libs.utils.cpu_memory import (CPUMemoryMonitor, clear_cpu_memory,
                                          log_comprehensive_memory_stats)
from shared_libs.utils.gpu_memory import clear_gpu_memory, log_gpu_memory_stats

logger = logging.getLogger(__name__)

class TrainingMemoryProfiler:
    """Comprehensive memory profiler for training processes."""

    def __init__(self, log_file: str = "memory_profile.jsonl", interval: float = 30.0):
        """
        Initialize memory profiler.

        Args:
            log_file: File to log memory statistics
            interval: Interval between memory checks in seconds
        """
        self.log_file = Path(log_file)
        self.interval = interval
        self.cpu_monitor = CPUMemoryMonitor()
        self.is_profiling = False
        self.start_time = None
        self.snapshots: List[Dict[str, Any]] = []

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Memory profiler initialized - logging to {log_file}")

    def start_profiling(self):
        """Start memory profiling."""
        self.is_profiling = True
        self.start_time = time.time()
        self.snapshots = []

        logger.info("Memory profiling started")

        # Take initial snapshot
        self._take_snapshot("profiling_start")

    def stop_profiling(self):
        """Stop memory profiling and save results."""
        if not self.is_profiling:
            return

        self.is_profiling = False

        # Take final snapshot
        self._take_snapshot("profiling_end")

        # Save all snapshots
        self._save_snapshots()

        duration = time.time() - self.start_time
        logger.info(f"Memory profiling stopped - {len(self.snapshots)} snapshots taken over {duration:.1f}s")

    def _take_snapshot(self, context: str = ""):
        """Take a memory snapshot."""
        if not self.is_profiling:
            return

        timestamp = time.time()

        # Get comprehensive memory stats
        cpu_stats = self.cpu_monitor.get_memory_stats()
        gpu_stats = {}

        if torch.cuda.is_available():
            gpu_stats = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
            }

        # Get garbage collector stats
        gc_stats = {
            "objects": len(gc.get_objects()),
            "collections": gc.get_count(),
            "garbage_count": len(gc.garbage)
        }

        # Create snapshot
        snapshot = {
            "timestamp": timestamp,
            "context": context,
            "elapsed_time": timestamp - self.start_time if self.start_time else 0,
            "cpu_memory": cpu_stats,
            "gpu_memory": gpu_stats,
            "gc_stats": gc_stats
        }

        self.snapshots.append(snapshot)

        # Log to console
        cpu_current = cpu_stats["current"]
        logger.info(f"Memory Snapshot ({context}): CPU {cpu_current['rss_mb']:.1f}MB")

        if gpu_stats:
            logger.info(f"  GPU: {gpu_stats['allocated_mb']:.1f}MB allocated")

    def _save_snapshots(self):
        """Save all snapshots to file."""
        try:
            with open(self.log_file, 'w') as f:
                for snapshot in self.snapshots:
                    json.dump(snapshot, f)
                    f.write('\n')

            logger.info(f"Saved {len(self.snapshots)} memory snapshots to {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to save memory snapshots: {e}")

    def periodic_snapshot(self, context: str = ""):
        """Take a periodic snapshot if enough time has elapsed."""
        if not self.is_profiling:
            return

        current_time = time.time()
        if not hasattr(self, '_last_snapshot_time'):
            self._last_snapshot_time = current_time
            self._take_snapshot(context)
            return

        if current_time - self._last_snapshot_time >= self.interval:
            self._last_snapshot_time = current_time
            self._take_snapshot(context)

def analyze_memory_profile(log_file: str) -> Dict[str, Any]:
    """
    Analyze memory profile and provide insights.

    Args:
        log_file: Path to memory profile log file

    Returns:
        Dictionary with analysis results
    """
    log_path = Path(log_file)
    if not log_path.exists():
        return {"error": f"Log file {log_file} does not exist"}

    snapshots = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                snapshots.append(json.loads(line.strip()))
    except Exception as e:
        return {"error": f"Failed to read log file: {e}"}

    if not snapshots:
        return {"error": "No snapshots found in log file"}

    # Analyze CPU memory trend
    cpu_memory_values = [s["cpu_memory"]["current"]["rss_mb"] for s in snapshots]
    cpu_start = cpu_memory_values[0]
    cpu_end = cpu_memory_values[-1]
    cpu_increase = cpu_end - cpu_start
    cpu_increase_percent = (cpu_increase / cpu_start) * 100 if cpu_start > 0 else 0

    # Analyze GPU memory trend
    gpu_memory_values = []
    for s in snapshots:
        if s["gpu_memory"]:
            gpu_memory_values.append(s["gpu_memory"]["allocated_mb"])
        else:
            gpu_memory_values.append(0.0)

    gpu_start = gpu_memory_values[0]
    gpu_end = gpu_memory_values[-1]
    gpu_increase = gpu_end - gpu_start
    gpu_increase_percent = (gpu_increase / gpu_start) * 100 if gpu_start > 0 else 0

    # Find memory spikes
    cpu_spikes = []
    gpu_spikes = []

    for i, snapshot in enumerate(snapshots):
        cpu_mb = snapshot["cpu_memory"]["current"]["rss_mb"]
        if i > 0:
            prev_cpu = snapshots[i-1]["cpu_memory"]["current"]["rss_mb"]
            if cpu_mb - prev_cpu > 50:  # 50MB increase
                cpu_spikes.append({
                    "index": i,
                    "context": snapshot["context"],
                    "increase_mb": cpu_mb - prev_cpu
                })

        if snapshot["gpu_memory"]:
            gpu_mb = snapshot["gpu_memory"]["allocated_mb"]
            if i > 0 and snapshots[i-1]["gpu_memory"]:
                prev_gpu = snapshots[i-1]["gpu_memory"]["allocated_mb"]
                if gpu_mb - prev_gpu > 100:  # 100MB increase
                    gpu_spikes.append({
                        "index": i,
                        "context": snapshot["context"],
                        "increase_mb": gpu_mb - prev_gpu
                    })

    analysis = {
        "total_snapshots": len(snapshots),
        "duration_seconds": snapshots[-1]["elapsed_time"] - snapshots[0]["elapsed_time"],
        "cpu_memory": {
            "start_mb": cpu_start,
            "end_mb": cpu_end,
            "increase_mb": cpu_increase,
            "increase_percent": cpu_increase_percent,
            "spikes": cpu_spikes
        },
        "gpu_memory": {
            "start_mb": gpu_start,
            "end_mb": gpu_end,
            "increase_mb": gpu_increase,
            "increase_percent": gpu_increase_percent,
            "spikes": gpu_spikes
        },
        "recommendations": []
    }

    # Generate recommendations
    if cpu_increase_percent > 10:
        analysis["recommendations"].append("High CPU memory increase detected. Consider:")
        analysis["recommendations"].append("  - Reducing DataLoader workers")
        analysis["recommendations"].append("  - Disabling persistent workers")
        analysis["recommendations"].append("  - Adding periodic CPU memory cleanup")

    if gpu_increase_percent > 20:
        analysis["recommendations"].append("High GPU memory increase detected. Consider:")
        analysis["recommendations"].append("  - Reducing batch size")
        analysis["recommendations"].append("  - Adding torch.cuda.empty_cache() calls")
        analysis["recommendations"].append("  - Using gradient accumulation")

    if cpu_spikes:
        analysis["recommendations"].append(f"CPU memory spikes detected at {len(cpu_spikes)} points")
        analysis["recommendations"].append("  - Investigate memory allocation patterns")

    if gpu_spikes:
        analysis["recommendations"].append(f"GPU memory spikes detected at {len(gpu_spikes)} points")
        analysis["recommendations"].append("  - Check for tensor accumulation in training loop")

    return analysis

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Memory profiling for training")
    parser.add_argument("--log-file", default="memory_profile.jsonl", help="Memory log file")
    parser.add_argument("--interval", type=float, default=30.0, help="Snapshot interval in seconds")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing log file")
    parser.add_argument("--output", default="memory_analysis.json", help="Analysis output file")

    args = parser.parse_args()

    if args.analyze:
        # Analyze existing log file
        analysis = analyze_memory_profile(args.log_file)

        # Save analysis
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Print summary
        print("Memory Analysis Summary:")
        print(f"Duration: {analysis['duration_seconds']:.1f}s")
        print(f"CPU Memory: {analysis['cpu_memory']['start_mb']:.1f}MB -> {analysis['cpu_memory']['end_mb']:.1f}MB ({analysis['cpu_memory']['increase_percent']:.1f}%)")
        if analysis['gpu_memory']['start_mb'] > 0:
            print(f"GPU Memory: {analysis['gpu_memory']['start_mb']:.1f}MB -> {analysis['gpu_memory']['end_mb']:.1f}MB ({analysis['gpu_memory']['increase_percent']:.1f}%)")

        if analysis["recommendations"]:
            print("\nRecommendations:")
            for rec in analysis["recommendations"]:
                print(f"  {rec}")

    else:
        # Start profiling
        profiler = TrainingMemoryProfiler(args.log_file, args.interval)

        try:
            profiler.start_profiling()

            print("Memory profiling started. Press Ctrl+C to stop...")

            while True:
                profiler.periodic_snapshot("periodic")
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nStopping profiling...")
            profiler.stop_profiling()

            # Analyze the results
            analysis = analyze_memory_profile(args.log_file)
            print(f"\nAnalysis saved to {args.output}")

if __name__ == "__main__":
    main()
