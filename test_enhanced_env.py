#!/usr/bin/env python3
"""
Test script for the enhanced env_or_colab module with Google Cloud support.
This script demonstrates how the module detects different environments and manages credentials.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.env_or_colab import (
    get_environment_info, 
    is_running_in_gcp, 
    is_running_in_colab,
    load_env_or_colab
)

def test_environment_detection():
    """Test environment detection in different scenarios."""
    print("=" * 60)
    print("🧪 Testing Enhanced Environment Detection")
    print("=" * 60)
    
    # Current environment
    print("\n📍 Current Environment:")
    env_info = get_environment_info()
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔍 Environment Checks:")
    print(f"  Is Colab: {is_running_in_colab()}")
    print(f"  Is GCP: {is_running_in_gcp()}")
    
    # Test different scenarios by setting environment variables
    print("\n🎭 Testing Different Scenarios:")
    
    # Scenario 1: Simulate Cloud Run
    print("\n1️⃣  Simulating Cloud Run Environment:")
    original_k_service = os.environ.get('K_SERVICE')
    original_project = os.environ.get('GOOGLE_CLOUD_PROJECT')
    
    os.environ['K_SERVICE'] = 'test-service'
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'test-project-123'
    
    print(f"  Is GCP: {is_running_in_gcp()}")
    env_info_cr = get_environment_info()
    print(f"  Type: {env_info_cr['type']}")
    print(f"  Project: {env_info_cr['project_id']}")
    
    # Restore original values
    if original_k_service:
        os.environ['K_SERVICE'] = original_k_service
    else:
        os.environ.pop('K_SERVICE', None)
    if original_project:
        os.environ['GOOGLE_CLOUD_PROJECT'] = original_project
    else:
        os.environ.pop('GOOGLE_CLOUD_PROJECT', None)
    
    # Scenario 2: Simulate App Engine
    print("\n2️⃣  Simulating App Engine Environment:")
    os.environ['GAE_SERVICE'] = 'default'
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'my-app-engine-project'
    
    print(f"  Is GCP: {is_running_in_gcp()}")
    env_info_gae = get_environment_info()
    print(f"  Type: {env_info_gae['type']}")
    print(f"  Project: {env_info_gae['project_id']}")
    
    # Clean up
    os.environ.pop('GAE_SERVICE', None)
    os.environ.pop('GOOGLE_CLOUD_PROJECT', None)
    
    # Scenario 3: Test credential detection
    print("\n3️⃣  Testing Credential Detection:")
    print(f"  WandB Key Available: {bool(os.environ.get('WANDB_API_KEY'))}")
    print(f"  GCP Credentials: {bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))}")
    
    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        print(f"  Credentials Path: {creds_path}")
        print(f"  File Exists: {os.path.exists(creds_path) if creds_path else False}")

def test_full_environment_setup():
    """Test the full environment setup process."""
    print("\n" + "=" * 60)
    print("🚀 Testing Full Environment Setup")
    print("=" * 60)
    
    # Test the main function
    load_env_or_colab()
    
    print("\n✅ Environment setup completed!")

def show_usage_examples():
    """Show usage examples for different deployment scenarios."""
    print("\n" + "=" * 60)
    print("📖 Usage Examples")
    print("=" * 60)
    
    examples = {
        "Local Development": {
            "description": "Using .env file for credentials",
            "setup": [
                "1. Create .env file in project root",
                "2. Add WANDB_API_KEY=your_key",
                "3. Add GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json",
                "4. Run your script normally"
            ]
        },
        "Google Colab": {
            "description": "Using Colab userdata for secrets",
            "setup": [
                "1. Go to 🔑 (secrets) in Colab sidebar",
                "2. Add WANDB_API_KEY secret",
                "3. Add GOOGLE_APPLICATION_CREDENTIALS secret (JSON content)",
                "4. Enable 'Notebook access' for both secrets",
                "5. Run your script in Colab"
            ]
        },
        "Google Cloud Run": {
            "description": "Using service account and environment variables",
            "setup": [
                "1. Deploy with service account attached",
                "2. Set WANDB_API_KEY as environment variable",
                "3. GOOGLE_APPLICATION_CREDENTIALS handled automatically",
                "4. Optional: Set GOOGLE_CLOUD_PROJECT if needed"
            ]
        },
        "Google Kubernetes Engine": {
            "description": "Using Workload Identity",
            "setup": [
                "1. Configure Workload Identity for your cluster",
                "2. Create Kubernetes Service Account",
                "3. Link to Google Service Account",
                "4. Set WANDB_API_KEY via Secret or ConfigMap",
                "5. Deploy your workload"
            ]
        },
        "Google Compute Engine": {
            "description": "Using attached service account",
            "setup": [
                "1. Create VM with service account attached",
                "2. Set WANDB_API_KEY via metadata or startup script",
                "3. Credentials handled via metadata service",
                "4. Project ID detected automatically"
            ]
        }
    }
    
    for env_name, details in examples.items():
        print(f"\n🌟 {env_name}")
        print(f"   {details['description']}")
        for step in details['setup']:
            print(f"   {step}")

if __name__ == "__main__":
    test_environment_detection()
    test_full_environment_setup()
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("=" * 60)
