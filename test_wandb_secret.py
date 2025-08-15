#!/usr/bin/env python3
"""
Test script to check if WandB API key is being loaded from Secret Manager.
"""
import os
import sys
sys.path.append('src')

from utils.env_or_colab import load_env_or_colab

def test_wandb_secret():
    print("Testing WandB Secret Manager integration...")
    
    # Check environment before loading
    wandb_key_before = os.environ.get("WANDB_API_KEY")
    print(f"WANDB_API_KEY before load_env_or_colab(): {wandb_key_before}")
    
    # Load environment/secrets
    print("Calling load_env_or_colab()...")
    load_env_or_colab()
    
    # Check environment after loading
    wandb_key_after = os.environ.get("WANDB_API_KEY")
    print(f"WANDB_API_KEY after load_env_or_colab(): {wandb_key_after}")
    
    if wandb_key_after:
        print("✅ WandB API key successfully loaded from Secret Manager!")
        # Don't print the actual key for security
        print(f"Key length: {len(wandb_key_after)} characters")
    else:
        print("❌ WandB API key not found in environment variables")
    
    # Test WandB logger initialization
    print("\nTesting WandB logger initialization...")
    try:
        from train.wandb_logger import WandbLogger
        logger = WandbLogger()
        if hasattr(logger, 'wandb') and logger.wandb is not None:
            print("✅ WandB logger initialized successfully!")
        else:
            print("❌ WandB logger not available")
    except Exception as e:
        print(f"❌ Error initializing WandB logger: {e}")

if __name__ == "__main__":
    test_wandb_secret()
