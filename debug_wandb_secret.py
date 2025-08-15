#!/usr/bin/env python3
"""
Detailed test script to debug WandB Secret Manager integration.
"""
import os
import sys
import logging

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('src')

def test_environment_detection():
    """Test environment detection logic."""
    print("=== Testing Environment Detection ===")
    
    from utils.env_or_colab import is_running_in_colab, is_running_in_gcp
    
    print(f"Is running in Colab: {is_running_in_colab()}")
    print(f"Is running in GCP: {is_running_in_gcp()}")
    
    # Check for relevant environment variables
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
    print(f"GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")

def test_secret_manager_direct():
    """Test Secret Manager functionality directly."""
    print("\n=== Testing Secret Manager Direct Access ===")
    
    from utils.env_or_colab import get_secret_from_manager
    
    # Test with the secret name from config
    secret_value = get_secret_from_manager("wandb-api-key")
    if secret_value:
        print(f"✅ Retrieved 'wandb-api-key' from Secret Manager (length: {len(secret_value)})")
    else:
        print("❌ Failed to retrieve 'wandb-api-key' from Secret Manager")
        
    # Test alternative name
    secret_value2 = get_secret_from_manager("WANDB_API_KEY")
    if secret_value2:
        print(f"✅ Retrieved 'WANDB_API_KEY' from Secret Manager (length: {len(secret_value2)})")
    else:
        print("❌ Failed to retrieve 'WANDB_API_KEY' from Secret Manager")

def test_config_loading():
    """Test configuration loading."""
    print("\n=== Testing Configuration Loading ===")
    
    from utils.env_or_colab import CONFIG
    
    print(f"Secrets config: {CONFIG.get('secrets', {})}")
    
    wandb_secrets = CONFIG.get('secrets', {}).get('WANDB_API_KEY', [])
    print(f"WandB secret names: {wandb_secrets}")

def test_full_environment_setup():
    """Test the full environment setup."""
    print("\n=== Testing Full Environment Setup ===")
    
    # Clear any existing WANDB_API_KEY
    if 'WANDB_API_KEY' in os.environ:
        del os.environ['WANDB_API_KEY']
        print("Cleared existing WANDB_API_KEY")
    
    # Import and run the setup
    from utils.env_or_colab import load_env_or_colab
    
    print("Running load_env_or_colab()...")
    load_env_or_colab()
    
    # Check the result
    wandb_key = os.environ.get('WANDB_API_KEY')
    if wandb_key:
        print(f"✅ WANDB_API_KEY set successfully (length: {len(wandb_key)})")
    else:
        print("❌ WANDB_API_KEY not set")

def test_environment_info():
    """Test environment info retrieval."""
    print("\n=== Testing Environment Info ===")
    
    from utils.env_or_colab import get_environment_info
    
    env_info = get_environment_info()
    for key, value in env_info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print("Starting detailed WandB Secret Manager debugging...\n")
    
    test_environment_detection()
    test_config_loading()
    test_secret_manager_direct()
    test_full_environment_setup()
    test_environment_info()
    
    print("\nDebugging complete!")
