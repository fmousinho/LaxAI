"""
Memory Pattern Documentation for WandB Checkpoint Saves

This test documents the expected memory behavior for WandB checkpoint operations:

1. Initial memory increase during WandB initialization (50-100MB) - ACCEPTABLE
2. Small memory increase during first checkpoint save (initialization costs) - ACCEPTABLE  
3. Stable memory usage for subsequent checkpoint saves (<5MB growth) - EXPECTED
4. No continuous memory growth after second checkpoint - REQUIRED

Memory growth from epoch 1->2 is acceptable (initialization), but not from epoch 2 onwards.
"""

import pytest
import time
import gc
import psutil
import torch
from unittest.mock import patch, MagicMock

from train.wandb_logger import wandb_logger
from utils.env_secrets import setup_environment_secrets


@pytest.mark.performance
def test_wandb_memory_pattern_documentation():
    """Document the expected memory pattern for WandB checkpoint saves.
    
    This test serves as documentation of the acceptable memory behavior:
    - One-time initialization cost is acceptable
    - Memory should stabilize after the first 1-2 checkpoint saves
    - No continuous growth in subsequent saves
    """
    setup_environment_secrets()
    process = psutil.Process()
    
    # Baseline memory
    gc.collect()
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    print(f"\n📊 Memory Pattern Analysis:")
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    
    with patch('wandb.init'), \
         patch('wandb.Artifact') as mock_artifact:
        
        mock_artifact_instance = MagicMock()
        mock_artifact.return_value = mock_artifact_instance
        
        # Create a mock run and attach it to wandb_logger  
        mock_run = MagicMock()
        mock_run.log_artifact = MagicMock()
        wandb_logger.run = mock_run
        wandb_logger.wandb_api = MagicMock()
        
        try:
            # Initialize WandB
            wandb_logger.init_run(config={'test': True}, run_name="memory_pattern_test")
            gc.collect()
            init_memory = process.memory_info().rss / (1024 * 1024)
            init_increase = init_memory - baseline_memory
            print(f"After WandB init: {init_memory:.1f} MB (+{init_increase:.1f} MB) ✅ ACCEPTABLE")
            
            # First checkpoint save
            state_dict = {'layer': torch.randn(1000, 1000)}  # ~4MB
            wandb_logger.save_checkpoint(epoch=1, model_state_dict=state_dict, optimizer_state_dict={}, loss=1.0)
            gc.collect()
            time.sleep(0.1)
            first_checkpoint_memory = process.memory_info().rss / (1024 * 1024)
            first_increase = first_checkpoint_memory - init_memory
            print(f"After 1st checkpoint: {first_checkpoint_memory:.1f} MB (+{first_increase:.1f} MB) ✅ ACCEPTABLE")
            
            # Second checkpoint - this is where we check for stabilization
            wandb_logger.save_checkpoint(epoch=2, model_state_dict=state_dict, optimizer_state_dict={}, loss=0.8)
            gc.collect()
            time.sleep(0.1)
            second_checkpoint_memory = process.memory_info().rss / (1024 * 1024)
            second_increase = second_checkpoint_memory - first_checkpoint_memory
            print(f"After 2nd checkpoint: {second_checkpoint_memory:.1f} MB (+{second_increase:.1f} MB)")
            
            # Additional checkpoints - should show stable memory
            memory_progression = [second_checkpoint_memory]
            for epoch in range(3, 6):
                wandb_logger.save_checkpoint(epoch=epoch, model_state_dict=state_dict, optimizer_state_dict={}, loss=1.0/epoch)
                gc.collect()
                time.sleep(0.1)
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_progression.append(current_memory)
                increase = current_memory - memory_progression[-2]
                print(f"After checkpoint {epoch}: {current_memory:.1f} MB (+{increase:.1f} MB)")
            
            # Analysis
            final_memory = memory_progression[-1]
            total_growth_after_stabilization = final_memory - second_checkpoint_memory
            
            print(f"\n📈 Memory Growth Analysis:")
            print(f"Initialization cost: {init_increase:.1f} MB ✅")
            print(f"First checkpoint cost: {first_increase:.1f} MB ✅")  
            print(f"Growth after stabilization (epochs 2-5): {total_growth_after_stabilization:.1f} MB")
            
            # Assertions for expected behavior
            assert init_increase < 150, f"WandB initialization used {init_increase:.1f} MB (threshold: 150 MB)"
            assert first_increase < 50, f"First checkpoint used {first_increase:.1f} MB (threshold: 50 MB)"
            assert total_growth_after_stabilization < 10, f"Memory continued growing by {total_growth_after_stabilization:.1f} MB after stabilization (threshold: 10 MB)"
            
            print(f"\n✅ Memory pattern is ACCEPTABLE - no memory leak detected!")
            print(f"   • One-time costs: {init_increase + first_increase:.1f} MB")
            print(f"   • Stable operation: {total_growth_after_stabilization:.1f} MB growth over 3 additional checkpoints")
            
        finally:
            try:
                wandb_logger.finish()
            except Exception:
                pass


if __name__ == "__main__":
    test_wandb_memory_pattern_documentation()
