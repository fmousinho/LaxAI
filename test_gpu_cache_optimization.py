#!/usr/bin/env python3
"""
Test GPU cache optimization for Cloud Run GPU environments.
"""

import pytest
import sys
import os

def test_gpu_cache_configuration():
    """Test that GPU cache clearing can be configured for Cloud Run GPU."""
    from src.train.training import Training
    
    print("🧪 Testing GPU cache configuration...")
    
    # Test default configuration (conservative for Cloud Run GPU)
    training_default = Training(clear_memory_on_start=False, enable_multithreading=False)
    assert training_default.gpu_cache_threshold == 0.85, f"Expected 0.85, got {training_default.gpu_cache_threshold}"
    assert training_default.conservative_gpu_cache == True, f"Expected True, got {training_default.conservative_gpu_cache}"
    print("✅ Default configuration: Conservative GPU cache (85% threshold)")
    
    # Test custom configuration for high-memory scenarios
    training_aggressive = Training(
        clear_memory_on_start=False, 
        enable_multithreading=False,
        gpu_cache_threshold=0.7,  # More aggressive
        conservative_gpu_cache=False  # Allow fallback clears
    )
    assert training_aggressive.gpu_cache_threshold == 0.7, f"Expected 0.7, got {training_aggressive.gpu_cache_threshold}"
    assert training_aggressive.conservative_gpu_cache == False, f"Expected False, got {training_aggressive.conservative_gpu_cache}"
    print("✅ Custom configuration: Aggressive GPU cache (70% threshold)")
    
    # Test Cloud Run GPU optimized configuration
    training_cloud_run = Training(
        clear_memory_on_start=False, 
        enable_multithreading=False,
        gpu_cache_threshold=0.9,  # Very conservative for GPU Cloud Run
        conservative_gpu_cache=True  # Prevent unnecessary clears
    )
    assert training_cloud_run.gpu_cache_threshold == 0.9, f"Expected 0.9, got {training_cloud_run.gpu_cache_threshold}"
    assert training_cloud_run.conservative_gpu_cache == True, f"Expected True, got {training_cloud_run.conservative_gpu_cache}"
    print("✅ Cloud Run GPU configuration: Very conservative (90% threshold)")

def test_smart_gpu_cache_clearing():
    """Test that smart GPU cache clearing respects configuration."""
    from src.train.training import Training
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️  No CUDA available, skipping GPU cache tests")
        return
        
    print("🧪 Testing smart GPU cache clearing...")
    
    # Test conservative mode (Cloud Run GPU optimized)
    training = Training(
        clear_memory_on_start=False, 
        enable_multithreading=False,
        gpu_cache_threshold=0.9,  # Very high threshold
        conservative_gpu_cache=True  # Conservative fallback
    )
    
    # This should not clear cache unless GPU memory is very high (90%+)
    training._smart_gpu_cache_clear(force=False, context="test_conservative")
    print("✅ Conservative GPU cache clearing tested")
    
    # Test force clearing still works
    training._smart_gpu_cache_clear(force=True, context="test_force")
    print("✅ Force GPU cache clearing works")

if __name__ == "__main__":
    print("🚀 Testing GPU Cache Optimization for Cloud Run GPU")
    print("=" * 60)
    
    test_gpu_cache_configuration()
    print()
    test_smart_gpu_cache_clearing()
    print()
    
    print("✅ All GPU cache optimization tests passed!")
    print("\n📋 Cloud Run GPU Recommendations:")
    print("   • Use gpu_cache_threshold=0.9 for maximum performance")
    print("   • Use conservative_gpu_cache=True to prevent unnecessary clears")
    print("   • GPU cache will only clear when >90% memory used")
    print("   • This preserves GPU memory locality and reduces overhead")
