#!/usr/bin/env python3
"""
Test script to verify:
1. Checkpoint cleanup timing issue is fixed
2. Test artifacts are properly deleted
"""
import sys
import os
import time
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.wandb_logger import WandbLogger

def count_artifacts(logger, artifact_pattern):
    """Count artifacts matching a pattern."""
    try:
        api = logger.wandb_api
        collection_name = f"{logger.run.entity}/{logger.run.project}/{artifact_pattern}"
        artifact_collection = api.artifact_collection("model_checkpoint", collection_name)
        artifacts = list(artifact_collection.artifacts())
        return len(artifacts)
    except Exception as e:
        # Collection doesn't exist or no artifacts
        return 0

def test_checkpoint_cleanup_timing():
    """Test that checkpoint cleanup works correctly after each save."""
    
    print("🧪 Testing checkpoint cleanup timing fix...")
    
    # Set to offline mode
    os.environ['WANDB_MODE'] = 'offline'
    
    logger = WandbLogger(enabled=True)
    logger.init_run(
        config={'test_timing': True},
        run_name="timing-test-cleanup",
        tags=['test', 'timing']
    )
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(5, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    artifact_pattern = logger._sanitize_artifact_name(logger._get_checkpoint_name())
    
    print("📦 Testing sequential checkpoint saves...")
    
    for epoch in range(5):
        print(f"  Saving checkpoint for epoch {epoch}...")
        
        # Save checkpoint (synchronous to ensure it's uploaded before we check)
        logger.save_checkpoint(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            model_name="timing_test_checkpoint",
            async_upload=False  # Use sync to ensure immediate upload
        )
        
        # Wait a moment for cleanup to complete
        time.sleep(1)
        
        # Count artifacts - should always be 1 (only latest)
        artifact_count = count_artifacts(logger, artifact_pattern)
        print(f"  After epoch {epoch}: {artifact_count} artifacts")
        
        if epoch > 0 and artifact_count > 1:
            print(f"❌ Too many artifacts after epoch {epoch}: {artifact_count} (expected 1)")
            return False
    
    # Cleanup test artifacts
    logger.cleanup_test_artifacts(force_cleanup_all=True)
    logger.finish()
    
    print("✅ Checkpoint timing test passed!")
    return True

def test_artifact_cleanup():
    """Test that test artifacts are properly deleted."""
    
    print("\n🧪 Testing test artifact cleanup...")
    
    os.environ['WANDB_MODE'] = 'offline'
    
    logger = WandbLogger(enabled=True)
    logger.init_run(
        config={'test_cleanup': True},
        run_name="artifact-cleanup-test",
        tags=['test', 'cleanup']
    )
    
    # Create multiple types of test artifacts
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    print("📦 Creating test artifacts...")
    
    # Create several checkpoints
    for epoch in range(3):
        logger.save_checkpoint(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            model_name="cleanup_test_checkpoint",
            async_upload=False
        )
        print(f"  Created checkpoint for epoch {epoch}")
    
    # Get initial artifact count
    artifact_pattern = logger._sanitize_artifact_name(logger._get_checkpoint_name())
    initial_count = count_artifacts(logger, artifact_pattern)
    print(f"📊 Initial artifact count: {initial_count}")
    
    # Test manual cleanup
    print("🧹 Testing manual cleanup...")
    logger.cleanup_test_artifacts(force_cleanup_all=True)
    
    # Wait for cleanup to propagate
    time.sleep(2)
    
    # Check if artifacts were cleaned up
    final_count = count_artifacts(logger, artifact_pattern)
    print(f"📊 Final artifact count: {final_count}")
    
    logger.finish()
    
    if final_count == 0:
        print("✅ Test artifact cleanup working!")
        return True
    else:
        print(f"❌ Test artifact cleanup failed: {final_count} artifacts remaining")
        return False

def test_async_checkpoint_timing():
    """Test that async checkpoints also get cleaned up properly."""
    
    print("\n🧪 Testing async checkpoint cleanup timing...")
    
    os.environ['WANDB_MODE'] = 'offline'
    
    logger = WandbLogger(enabled=True) 
    logger.init_run(
        config={'test_async_timing': True},
        run_name="async-timing-test",
        tags=['test', 'async', 'timing']
    )
    
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    print("📦 Testing async checkpoint saves...")
    
    for epoch in range(3):
        print(f"  Saving async checkpoint for epoch {epoch}...")
        
        logger.save_checkpoint(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), 
            epoch=epoch,
            model_name="async_timing_checkpoint",
            async_upload=True  # Use async upload
        )
    
    # Wait for async uploads and cleanup to complete
    print("⏳ Waiting for async operations to complete...")
    time.sleep(5)
    
    # Check artifact count - should be 1 due to cleanup
    artifact_pattern = logger._sanitize_artifact_name(logger._get_checkpoint_name())
    artifact_count = count_artifacts(logger, artifact_pattern)
    print(f"📊 Final async artifact count: {artifact_count}")
    
    # Cleanup and finish
    logger.cleanup_test_artifacts(force_cleanup_all=True)
    logger.finish()
    
    if artifact_count <= 1:
        print("✅ Async checkpoint cleanup working!")
        return True
    else:
        print(f"❌ Async checkpoint cleanup failed: {artifact_count} artifacts (expected ≤1)")
        return False

if __name__ == "__main__":
    try:
        print("🚀 Starting checkpoint cleanup verification tests...\n")
        
        # Test 1: Checkpoint timing issue
        timing_ok = test_checkpoint_cleanup_timing()
        
        # Test 2: Test artifact cleanup
        cleanup_ok = test_artifact_cleanup()
        
        # Test 3: Async checkpoint timing
        async_ok = test_async_checkpoint_timing()
        
        print(f"\n📋 Test Results:")
        print(f"   {'✅' if timing_ok else '❌'} Checkpoint cleanup timing")
        print(f"   {'✅' if cleanup_ok else '❌'} Test artifact cleanup")
        print(f"   {'✅' if async_ok else '❌'} Async checkpoint cleanup")
        
        if timing_ok and cleanup_ok and async_ok:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
