#!/usr/bin/env python3
"""
Test to demonstrate checkpoint cleanup issue.
"""
import sys
import os
import time
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.wandb_logger import WandbLogger

def test_checkpoint_cleanup_issue():
    """Test that demonstrates the checkpoint cleanup issue."""
    
    print("🧪 Testing checkpoint cleanup behavior...")
    
    # Set to offline mode
    os.environ['WANDB_MODE'] = 'offline'
    
    logger = WandbLogger(enabled=True)
    logger.init_run(
        config={'cleanup_test': True},
        run_name="checkpoint-cleanup-test",
        tags=['test', 'checkpoint-cleanup']
    )
    
    # Create test model and optimizer
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    print("📦 Saving multiple checkpoints sequentially...")
    
    # Save multiple checkpoints to see if cleanup happens
    for epoch in range(1, 6):
        print(f"   Saving checkpoint for epoch {epoch}...")
        
        logger.save_checkpoint(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            model_name="cleanup_test",
            async_upload=False  # Use sync to avoid timing issues
        )
        
        # Small delay to allow wandb to process
        time.sleep(1)
        
        # Check if artifacts are being cleaned up by monitoring the API
        try:
            api = logger.wandb_api
            if api:
                artifact_name = logger._sanitize_artifact_name(logger._get_checkpoint_name())
                artifact_type_api = api.artifact_type("model_checkpoint", project="LaxAI")
                artifact_collection = artifact_type_api.collection(artifact_name)
                
                artifacts = list(artifact_collection.artifacts())
                print(f"   After epoch {epoch}: Found {len(artifacts)} artifacts in collection")
                
                if len(artifacts) > 1:
                    print(f"   ⚠️  Expected 1 artifact (keep_latest=1), but found {len(artifacts)}")
                    for i, art in enumerate(artifacts):
                        print(f"      {i+1}. {art.name} (created: {getattr(art, 'created_at', 'unknown')})")
                else:
                    print(f"   ✅ Cleanup working: only {len(artifacts)} artifact(s) found")
                    
        except Exception as e:
            print(f"   Could not check artifacts: {e}")
    
    print("🧹 Finishing logger...")
    logger.finish()
    
    return True

if __name__ == "__main__":
    try:
        test_checkpoint_cleanup_issue()
        print("✅ Test completed")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
