#!/usr/bin/env python3
"""
Quick test to demonstrate that test artifacts are now properly cleaned up.
"""
import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.wandb_logger import WandbLogger

def main():
    print("🧪 Testing improved wandb cleanup...")
    
    # Set to offline mode
    os.environ['WANDB_MODE'] = 'offline'
    
    # Create a test run with cleanup
    logger = WandbLogger(enabled=True)
    logger.init_run(
        config={'demo': True},
        run_name="demo-test-cleanup",  # Contains 'test' - will auto-cleanup
        tags=['test', 'demo']
    )
    
    # Create test artifacts
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    print("📦 Creating test artifacts...")
    logger.save_checkpoint(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=1,
        model_name="demo_checkpoint"
    )
    
    print("🧹 Finishing with auto-cleanup...")
    logger.finish()  # Will auto-detect test run and cleanup
    
    print("✅ Test completed - artifacts should be cleaned up automatically!")

if __name__ == "__main__":
    main()
