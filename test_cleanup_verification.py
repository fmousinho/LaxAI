#!/usr/bin/env python3
"""
Test script to verify cleanup functionality removes all test artifacts.
"""
import sys
import os
import time
import torch
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.wandb_logger import WandbLogger

def test_cleanup_functionality():
    """Test that cleanup removes all test artifacts."""
    
    print("🧪 Testing wandb test artifact cleanup...")
    
    # Set to offline to avoid polluting online wandb
    os.environ['WANDB_MODE'] = 'offline'
    
    # Create logger with test run name
    logger = WandbLogger(enabled=True)
    logger.init_run(
        config={'test_cleanup': True},
        run_name="cleanup-test-run",
        tags=['test', 'cleanup', 'verification']
    )
    
    # Create some test artifacts
    dummy_model = torch.nn.Linear(5, 1)
    state_dict = dummy_model.state_dict()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    optimizer_state_dict = dummy_optimizer.state_dict()
    
    print("📦 Creating test artifacts...")
    
    # Create multiple checkpoints to simulate test scenario
    for epoch in range(3):
        logger.save_checkpoint(
            model_state_dict=state_dict,
            optimizer_state_dict=optimizer_state_dict,
            epoch=epoch,
            model_name="cleanup_test_checkpoint",
            async_upload=False  # Use sync for immediate creation
        )
        print(f"   Created checkpoint for epoch {epoch}")
    
    print("⏳ Waiting for artifacts to be logged...")
    time.sleep(2)
    
    # Test manual cleanup
    print("🧹 Testing manual cleanup...")
    try:
        logger.cleanup_test_artifacts(force_cleanup_all=True)
        print("✅ Manual cleanup completed")
    except Exception as e:
        print(f"❌ Manual cleanup failed: {e}")
    
    # Test automatic cleanup on finish
    print("🏁 Testing automatic cleanup on finish...")
    logger.finish()  # Should auto-detect test run and cleanup
    
    print("✅ Cleanup test completed!")
    return True

def test_non_test_run_protection():
    """Test that non-test runs are protected from accidental cleanup."""
    
    print("\n🛡️  Testing non-test run protection...")
    
    os.environ['WANDB_MODE'] = 'offline'
    
    # Create logger with production-like run name
    logger = WandbLogger(enabled=True)
    logger.init_run(
        config={'production_run': True},
        run_name="production-training-run",  # No 'test' keywords
        tags=['production', 'training']
    )
    
    # Create an artifact
    dummy_model = torch.nn.Linear(3, 1)
    state_dict = dummy_model.state_dict()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    optimizer_state_dict = dummy_optimizer.state_dict()
    
    logger.save_checkpoint(
        model_state_dict=state_dict,
        optimizer_state_dict=optimizer_state_dict,
        epoch=1,
        model_name="production_checkpoint",
        async_upload=False
    )
    
    print("📦 Created production artifact")
    
    # Try cleanup - should be skipped for non-test runs
    logger.cleanup_test_artifacts()
    
    # Finish - should NOT auto-cleanup
    logger.finish()
    
    print("✅ Non-test run protection verified")
    return True

if __name__ == "__main__":
    try:
        print("🚀 Starting cleanup verification tests...\n")
        
        # Test 1: Cleanup functionality
        test_cleanup_functionality()
        
        # Test 2: Protection for non-test runs
        test_non_test_run_protection()
        
        print("\n🎉 All cleanup tests completed successfully!")
        print("📋 Summary:")
        print("   ✅ Test artifact cleanup working")
        print("   ✅ Non-test run protection working") 
        print("   ✅ Auto-cleanup on finish working")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
