#!/usr/bin/env python3
"""
Test script for Google Secret Manager integration in env_or_colab.py

This script demonstrates how to:
1. Check Secret Manager availability
2. Retrieve secrets from Secret Manager
3. Test the integrated credential setup

Prerequisites:
- google-cloud-secret-manager installed
- GCP credentials configured (service account or default credentials)
- Secrets created in Secret Manager (e.g., 'wandb-api-key')

To create a secret in Secret Manager:
```bash
# Enable the Secret Manager API
gcloud services enable secretmanager.googleapis.com

# Create a secret
gcloud secrets create wandb-api-key --data-file=-
# (then paste your WandB API key and press Ctrl+D)

# Or create from command line
echo "your-wandb-api-key-here" | gcloud secrets create wandb-api-key --data-file=-
```
"""

import os
import sys
import logging

# Add the src directory to the path so we can import our modules
sys.path.insert(0, 'src')

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_secret_manager_availability():
    """Test if Secret Manager client is available."""
    print("🔍 Testing Secret Manager availability...")
    try:
        from google.cloud import secretmanager
        print("✅ Secret Manager client is available")
        return True
    except ImportError as e:
        print(f"❌ Secret Manager client not available: {e}")
        return False

def test_secret_retrieval():
    """Test retrieving a secret from Secret Manager."""
    print("\n🔐 Testing secret retrieval...")
    
    # Import our module
    from utils.env_or_colab import get_secret_from_manager
    
    # Test with a common secret name
    secret_names = ['wandb-api-key', 'WANDB_API_KEY', 'test-secret']
    
    for secret_name in secret_names:
        print(f"   Trying to retrieve secret: {secret_name}")
        secret_value = get_secret_from_manager(secret_name)
        if secret_value:
            # Don't print the actual secret value for security
            print(f"   ✅ Successfully retrieved '{secret_name}' (length: {len(secret_value)})")
            return True
        else:
            print(f"   ❌ Could not retrieve '{secret_name}'")
    
    print("   ⚠️  No secrets found. Make sure you have created secrets in Secret Manager.")
    return False

def test_environment_setup():
    """Test the full environment setup with Secret Manager."""
    print("\n🚀 Testing environment setup with Secret Manager...")
    
    # Clear existing environment variables to test Secret Manager fallback
    original_wandb_key = os.environ.get('WANDB_API_KEY')
    if original_wandb_key:
        del os.environ['WANDB_API_KEY']
        print("   Temporarily cleared WANDB_API_KEY to test Secret Manager")
    
    try:
        # Import and run the environment setup
        from utils.env_or_colab import load_env_or_colab, get_environment_info
        
        # Reload the environment
        load_env_or_colab()
        
        # Get environment info
        env_info = get_environment_info()
        
        print("\n📊 Environment Information:")
        for key, value in env_info.items():
            print(f"   {key}: {value}")
        
        # Check if WandB key was loaded
        if os.environ.get('WANDB_API_KEY'):
            print("\n✅ WANDB_API_KEY successfully loaded from Secret Manager!")
        else:
            print("\n⚠️  WANDB_API_KEY not loaded. Check if 'wandb-api-key' secret exists.")
        
    finally:
        # Restore original environment
        if original_wandb_key:
            os.environ['WANDB_API_KEY'] = original_wandb_key
            print("   Restored original WANDB_API_KEY")

def test_gcp_project_detection():
    """Test GCP project detection for Secret Manager."""
    print("\n🌤️  Testing GCP project detection...")
    
    project_sources = [
        ('GOOGLE_CLOUD_PROJECT', os.environ.get('GOOGLE_CLOUD_PROJECT')),
        ('GCLOUD_PROJECT', os.environ.get('GCLOUD_PROJECT')),
        ('GCP_PROJECT', os.environ.get('GCP_PROJECT')),
    ]
    
    found_project = False
    for source, value in project_sources:
        if value:
            print(f"   ✅ Project ID found in {source}: {value}")
            found_project = True
            break
    
    if not found_project:
        print("   ⚠️  No project ID found in environment variables")
        print("   Trying metadata service...")
        
        try:
            import requests
            response = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/project/project-id',
                headers={'Metadata-Flavor': 'Google'},
                timeout=2
            )
            if response.status_code == 200:
                project_id = response.text
                print(f"   ✅ Project ID from metadata service: {project_id}")
                found_project = True
        except Exception as e:
            print(f"   ❌ Metadata service not accessible: {e}")
    
    if not found_project:
        print("   ❌ No GCP project detected. Set GOOGLE_CLOUD_PROJECT or run in GCP.")

def main():
    """Run all tests."""
    print("🧪 Testing Google Secret Manager Integration")
    print("=" * 50)
    
    # Test 1: Check availability
    if not test_secret_manager_availability():
        print("\n❌ Cannot proceed without Secret Manager client.")
        print("Install with: pip install google-cloud-secret-manager")
        return
    
    # Test 2: Project detection
    test_gcp_project_detection()
    
    # Test 3: Secret retrieval
    test_secret_retrieval()
    
    # Test 4: Full environment setup
    test_environment_setup()
    
    print("\n" + "=" * 50)
    print("🏁 Secret Manager integration test complete!")
    print("\nTo use Secret Manager in your application:")
    print("1. Create secrets: gcloud secrets create wandb-api-key --data-file=-")
    print("2. Grant access: gcloud secrets add-iam-policy-binding wandb-api-key \\")
    print("   --member='serviceAccount:your-service-account@project.iam.gserviceaccount.com' \\")
    print("   --role='roles/secretmanager.secretAccessor'")
    print("3. The env_or_colab module will automatically try Secret Manager as fallback")

if __name__ == "__main__":
    main()
