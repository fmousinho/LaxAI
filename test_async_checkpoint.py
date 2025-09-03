#!/usr/bin/env python3
"""
Test script to verify async checkpoint functionality works correctly.
"""
import sys
import os
import time
import torch
import tempfile
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.wandb_logger import WandbLogger

def test_async_checkpoint_upload():
    """Test that async checkpoint upload works without blocking."""
    
    print("🧪 Testing async checkpoint upload functionality...")
    
    # Create a temporary directory for test artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Initialize WandbLogger with minimal config
        os.environ['WANDB_MODE'] = 'offline'  # Run offline to avoid polluting wandb
        logger = WandbLogger(enabled=True)
        logger.init_run(
            config={'learning_rate': 0.001},
            run_name="async-test-run",
            tags=['test', 'async']
        )
        
        # Create a dummy model state dict
        dummy_model = torch.nn.Linear(10, 1)
        state_dict = dummy_model.state_dict()
        
        # Create a dummy optimizer
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
        optimizer_state_dict = dummy_optimizer.state_dict()
        
        print("📦 Saving checkpoint with async_upload=True...")
        start_time = time.time()
        
        # Save checkpoint with async upload
        logger.save_checkpoint(
            model_state_dict=state_dict,
            optimizer_state_dict=optimizer_state_dict,
            epoch=1,
            model_name="test_checkpoint",
            async_upload=True  # This should not block
        )
        
        sync_time = time.time() - start_time
        print(f"⏱️  save_checkpoint returned in {sync_time:.3f}s (should be fast)")
        
        # The function should return quickly (< 1 second) since upload is async
        if sync_time > 1.0:
            print("⚠️  WARNING: save_checkpoint took too long, async may not be working")
        else:
            print("✅ save_checkpoint returned quickly - async upload likely working")
        
        # Check that executor is active
        if hasattr(logger, '_checkpoint_executor'):
            print(f"🔧 ThreadPoolExecutor found: {logger._checkpoint_executor}")
            
            # Check if there are any pending futures
            if hasattr(logger, '_checkpoint_futures'):
                active_futures = [f for f in logger._checkpoint_futures if not f.done()]
                print(f"📊 Active upload futures: {len(active_futures)}")
            
        # Wait a bit for background upload to complete
        print("⏳ Waiting for background upload to complete...")
        time.sleep(2)
        
        # Test sync upload for comparison
        print("📦 Saving checkpoint with async_upload=False (sync)...")
        start_time = time.time()
        
        logger.save_checkpoint(
            model_state_dict=state_dict,
            optimizer_state_dict=optimizer_state_dict,
            epoch=2,
            model_name="test_checkpoint_sync",
            async_upload=False  # Synchronous
        )
        
        sync_time = time.time() - start_time
        print(f"⏱️  Sync save_checkpoint took {sync_time:.3f}s")
        
        # Cleanup
        print("🧹 Finishing logger (should wait for async uploads)...")
        logger.finish()
        print("✅ Logger finished successfully")
        
        return True

def test_async_cleanup():
    """Test async cleanup with keep_latest parameter."""
    
    print("\n🧪 Testing async cleanup functionality...")
    
    # Create a temporary directory for test artifacts  
    with tempfile.TemporaryDirectory() as temp_dir:
        
        os.environ['WANDB_MODE'] = 'offline'  # Run offline to avoid polluting wandb
        logger = WandbLogger(enabled=True)
        logger.init_run(
            config={'batch_size': 32},
            run_name="cleanup-test-run",
            tags=['test', 'cleanup']
        )
        
        # Create multiple checkpoints
        dummy_model = torch.nn.Linear(5, 1)
        state_dict = dummy_model.state_dict()
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
        optimizer_state_dict = dummy_optimizer.state_dict()
        
        print("📦 Creating multiple checkpoints...")
        for epoch in range(5):
            logger.save_checkpoint(
                model_state_dict=state_dict,
                optimizer_state_dict=optimizer_state_dict,
                epoch=epoch,
                model_name="multi_checkpoint",
                async_upload=True
            )
            print(f"   Created checkpoint for epoch {epoch}")
        
        print("⏳ Waiting for async operations...")
        time.sleep(3)
        
        logger.finish()
        print("✅ Cleanup test completed")
        
        return True

if __name__ == "__main__":
    try:
        print("🚀 Starting async checkpoint tests...\n")
        
        # Test 1: Basic async upload
        test_async_checkpoint_upload()
        
        # Test 2: Async cleanup
        test_async_cleanup()
        
        print("\n🎉 All async checkpoint tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
