#!/usr/bin/env python3
"""
Test script for wandb checkpoint save/load functionality.

This script tests the checkpoint save and load features of the WandbLogger class.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from train.wandb_logger import WandbLogger
from config.all_config import wandb_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple test model for checkpoint testing."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_checkpoint_functionality():
    """Test the complete checkpoint save/load cycle."""
    
    logger.info("=" * 60)
    logger.info("Testing WandB Checkpoint Save/Load Functionality")
    logger.info("=" * 60)
    
    # Check if wandb is available
    try:
        import wandb
        logger.info("✓ wandb package is available")
    except ImportError:
        logger.error("✗ wandb package not available - cannot test checkpoint functionality")
        return False
    
    # Check API key
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        logger.error("✗ WANDB_API_KEY environment variable not found")
        logger.info("Please set your wandb API key: export WANDB_API_KEY=your_key_here")
        return False
    else:
        logger.info("✓ WANDB_API_KEY found")
    
    try:
        # Initialize wandb logger
        logger.info("\n1. Initializing WandB Logger...")
        wandb_logger = WandbLogger(enabled=True)
        
        if not wandb_logger.enabled:
            logger.error("✗ WandB logger not enabled")
            return False
        
        logger.info("✓ WandB logger initialized")
        
        # Initialize wandb run
        logger.info("\n2. Starting WandB run...")
        test_config = {
            "model_type": "SimpleTestModel",
            "input_size": 10,
            "hidden_size": 20,
            "output_size": 5,
            "test_run": True
        }
        
        success = wandb_logger.init_run(
            config=test_config,
            run_name="checkpoint_test_run",
            tags=["test", "checkpoint"]
        )
        
        if not success:
            logger.error("✗ Failed to initialize wandb run")
            return False
        
        logger.info("✓ WandB run initialized successfully")
        logger.info(f"   Run URL: {wandb_logger.run.url if wandb_logger.run else 'N/A'}")
        
        # Create test model and optimizer
        logger.info("\n3. Creating test model and optimizer...")
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few "epochs" to change model state
        dummy_input = torch.randn(5, 10)
        dummy_target = torch.randn(5, 5)
        criterion = nn.MSELoss()
        
        initial_loss = None
        for epoch in range(3):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
            
            logger.info(f"   Epoch {epoch + 1}: loss = {loss.item():.6f}")
        
        logger.info("✓ Model training completed")
        
        # Save checkpoint
        logger.info("\n4. Saving checkpoint to WandB...")
        model_name = "test_model"
        epoch_to_save = 3
        final_loss = loss.item()
        
        model_config = {
            "input_size": 10,
            "hidden_size": 20,
            "output_size": 5
        }
        
        checkpoint_path = wandb_logger.save_checkpoint(
            epoch=epoch_to_save,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            loss=final_loss,
            model_name=model_name,
            model_config=model_config
        )
        
        if checkpoint_path:
            logger.info(f"✓ Checkpoint saved successfully: {checkpoint_path}")
        else:
            logger.error("✗ Failed to save checkpoint")
            return False
        
        # Create a new model with different state
        logger.info("\n5. Creating new model to test loading...")
        new_model = SimpleTestModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        # Get initial state for comparison
        old_param = list(model.parameters())[0].data.clone()
        new_param_before = list(new_model.parameters())[0].data.clone()
        
        logger.info(f"   Original model first param sum: {old_param.sum().item():.6f}")
        logger.info(f"   New model first param sum (before load): {new_param_before.sum().item():.6f}")
        
        # Load checkpoint (with retry logic for artifact availability)
        logger.info("\n6. Loading checkpoint from WandB...")
        artifact_name = f"{model_name}_checkpoint"
        
        # Wait a moment for the artifact to be fully uploaded and available
        import time
        logger.info("   Waiting for artifact to be available...")
        time.sleep(5)
        
        # Try loading with retry logic
        checkpoint_data = None
        max_retries = 3
        for attempt in range(max_retries):
            logger.info(f"   Attempt {attempt + 1}/{max_retries} to load checkpoint...")
            checkpoint_data = wandb_logger.load_checkpoint(artifact_name, version="latest")
            if checkpoint_data:
                break
            if attempt < max_retries - 1:
                logger.info(f"   Waiting 3 seconds before retry...")
                time.sleep(3)
        
        if checkpoint_data:
            logger.info("✓ Checkpoint loaded successfully")
            logger.info(f"   Loaded epoch: {checkpoint_data.get('epoch', 'unknown')}")
            logger.info(f"   Loaded loss: {checkpoint_data.get('loss', 'unknown')}")
            logger.info(f"   Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
            
            # Verify checkpoint contents
            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss', 'model_config']
            missing_keys = [key for key in required_keys if key not in checkpoint_data]
            
            if missing_keys:
                logger.error(f"✗ Missing keys in checkpoint: {missing_keys}")
                return False
            
            logger.info("✓ All required keys present in checkpoint")
            
        else:
            logger.error("✗ Failed to load checkpoint")
            return False
        
        # Test resume functionality
        logger.info("\n7. Testing resume training functionality...")
        start_epoch = wandb_logger.resume_training_from_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            artifact_name=artifact_name,
            version="latest"
        )
        
        if start_epoch > 1:
            logger.info(f"✓ Resume training successful, starting from epoch {start_epoch}")
        else:
            logger.error("✗ Resume training failed")
            return False
        
        # Verify model states match
        new_param_after = list(new_model.parameters())[0].data.clone()
        logger.info(f"   New model first param sum (after load): {new_param_after.sum().item():.6f}")
        
        if torch.allclose(old_param, new_param_after, atol=1e-6):
            logger.info("✓ Model state successfully restored")
        else:
            logger.error("✗ Model state restoration failed - parameters don't match")
            return False
        
        # Test training metrics logging
        logger.info("\n8. Testing training metrics logging...")
        wandb_logger.log_training_metrics(
            epoch=start_epoch,
            train_loss=final_loss,
            val_loss=final_loss * 0.9,
            learning_rate=0.001
        )
        logger.info("✓ Training metrics logged successfully")
        
        # Test batch metrics logging
        wandb_logger.log_batch_metrics(
            batch_idx=10,
            epoch=start_epoch,
            batch_loss=final_loss * 1.1
        )
        logger.info("✓ Batch metrics logged successfully")
        
        # Finish wandb run
        logger.info("\n9. Finishing WandB run...")
        wandb_logger.finish()
        logger.info("✓ WandB run finished successfully")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL CHECKPOINT TESTS PASSED! 🎉")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with exception: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Main test function."""
    success = test_checkpoint_functionality()
    
    if success:
        print("\n✅ Checkpoint functionality is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Checkpoint functionality test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
