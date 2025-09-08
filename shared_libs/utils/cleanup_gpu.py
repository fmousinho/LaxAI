#!/usr/bin/env python3
"""
GPU Memory Cleanup Script

This script can be run independently to clear GPU memory,
useful for recovering from crashed training sessions.

Usage:
    python src/utils/cleanup_gpu.py
    python src/utils/cleanup_gpu.py --force
"""

import argparse
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.gpu_memory import clear_gpu_memory, log_gpu_memory_stats, get_gpu_memory_stats
import torch


def main():
    parser = argparse.ArgumentParser(description='Clean up GPU memory')
    parser.add_argument('--force', action='store_true', 
                       help='Perform aggressive cleanup including tensor search')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show memory statistics without cleanup')
    
    args = parser.parse_args()
    
    print("GPU Memory Cleanup Utility")
    print("=" * 40)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return
    
    print(f"Found {torch.cuda.device_count()} CUDA device(s)")
    
    # Show initial stats
    print("\nBefore cleanup:")
    log_gpu_memory_stats("Initial state")
    
    if not args.stats_only:
        # Perform cleanup
        print(f"\nPerforming {'aggressive' if args.force else 'standard'} cleanup...")
        clear_gpu_memory(force=args.force)
        
        # Show final stats
        print("\nAfter cleanup:")
        log_gpu_memory_stats("Final state")
    
    # Show detailed stats
    stats = get_gpu_memory_stats()
    print(f"\nDetailed Statistics:")
    print(f"  Allocated: {stats['allocated_gb']:.2f} GB ({stats['allocated_bytes']:,} bytes)")
    print(f"  Cached:    {stats['cached_gb']:.2f} GB ({stats['cached_bytes']:,} bytes)")
    print(f"  Peak:      {stats['max_allocated_gb']:.2f} GB ({stats['max_allocated_bytes']:,} bytes)")
    
    if stats['allocated_gb'] > 0.1:  # More than 100MB still allocated
        print(f"\nWarning: {stats['allocated_gb']:.2f} GB still allocated.")
        print("This might indicate active PyTorch processes or persistent allocations.")
        print("Try:")
        print("1. Ensuring all Python processes using PyTorch are terminated")
        print("2. Running with --force flag for more aggressive cleanup")
        print("3. Restarting your Python kernel/environment")


if __name__ == "__main__":
    main()
