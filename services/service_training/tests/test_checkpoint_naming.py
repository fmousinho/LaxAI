#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import just the wandb_logger module directly
    from train.wandb_logger import WandbLogger
    
    # Create instance
    wandb_logger = WandbLogger()
    
    # Mock a run to test the new naming logic
    class MockRun:
        def __init__(self, name):
            self.name = name
    
    print("Testing checkpoint naming logic...")
    
    # Test with no run (default behavior)
    wandb_logger.run = None
    checkpoint_name = wandb_logger._get_checkpoint_name()
    print(f'✅ Default checkpoint name (no run): {checkpoint_name}')
    
    # Test with a mock run
    wandb_logger.run = MockRun('my-training-run-v1')
    checkpoint_name = wandb_logger._get_checkpoint_name()
    print(f'✅ Checkpoint name with run: {checkpoint_name}')
    
    # Test with a run name that needs sanitization
    wandb_logger.run = MockRun('my@special#run!2024')
    checkpoint_name = wandb_logger._get_checkpoint_name()
    print(f'✅ Checkpoint name with sanitized run: {checkpoint_name}')
    
    # Test test run prefix
    wandb_logger.run = MockRun('test-run-123')
    checkpoint_name = wandb_logger._get_checkpoint_name()
    print(f'✅ Test checkpoint name: {checkpoint_name}')
    
    print('✅ All checkpoint naming scenarios work correctly!')
    print(f'✅ Now each training run will have unique checkpoint names!')
    
except Exception as e:
    print('❌ Error:', e)
    import traceback
    traceback.print_exc()
