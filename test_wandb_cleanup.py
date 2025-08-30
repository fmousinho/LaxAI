#!/usr/bin/env python3
"""
Test script to validate WandB cleanup functionality with corrected artifact path construction.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train.wandb_logger import WandbLogger
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_cleanup_functionality():
    """Test the _cleanup_old_checkpoints method with corrected path construction."""

    try:
        # Initialize WandB logger
        wandb_logger = WandbLogger()

        # Test with a sample artifact name
        test_artifact_name = "test_checkpoint_model"

        logger.info("Testing WandB cleanup functionality...")

        # This should work without the 'NoneType' object has no attribute 'artifacts' error
        wandb_logger._cleanup_old_checkpoints(test_artifact_name, keep_latest=1)

        logger.info("✅ Cleanup test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Cleanup test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = test_cleanup_functionality()
    sys.exit(0 if success else 1)
