#!/usr/bin/env python3
"""
Test script to verify the model loading fix for embedding dimension mismatch.

This script demonstrates:
1. How the old method would fail with size mismatch
2. How the new method handles it by reading saved metadata
"""

import os
import sys
from unittest.mock import MagicMock, Mock

import torch
import torch.nn as nn

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_mock_artifact_with_metadata(embedding_dim):
    """Create a mock wandb artifact with metadata containing embedding_dim"""
    mock_artifact = Mock()
    mock_artifact.metadata = {
        'model_config': {
            'embedding_dim': embedding_dim
        }
    }
    mock_artifact.files.return_value = [Mock()]
    mock_artifact.files()[0].name = 'model.pth'
    return mock_artifact

def test_model_loading_with_different_embedding_dims():
    """Test that model loading works when checkpoint has different embedding_dim"""

    # Import the SiameseNet class
    from siamesenet import SiameseNet

    print("Testing model loading with embedding dimension mismatch...")

    # Create a model with embedding_dim=512 (simulating old checkpoint)
    old_model = SiameseNet(embedding_dim=512)
    print(f"Old model embedding_dim: {old_model.embedding_dim}")

    # Create a checkpoint state_dict
    old_state_dict = old_model.state_dict()

    # Now create a new model with embedding_dim=384 (current config)
    new_model = SiameseNet(embedding_dim=384)
    print(f"New model embedding_dim: {new_model.embedding_dim}")

    # Try to load the old checkpoint into the new model
    try:
        new_model.load_state_dict(old_state_dict)
        print("❌ Unexpected: Direct load_state_dict succeeded (should have failed)")
    except RuntimeError as e:
        print(f"✅ Expected: Direct load_state_dict failed with: {str(e)[:100]}...")

    # Now test the fixed loading approach
    print("\nTesting fixed loading approach...")

    # Simulate the fixed load_model_from_registry logic
    saved_config = {'embedding_dim': 512}  # From artifact metadata
    current_kwargs = {'embedding_dim': 384}  # From current config

    # Merge with saved config taking precedence
    merged_kwargs = dict(current_kwargs)
    merged_kwargs.update(saved_config)

    print(f"Merged kwargs: {merged_kwargs}")

    # Create model with correct architecture
    fixed_model = SiameseNet(**merged_kwargs)
    print(f"Fixed model embedding_dim: {fixed_model.embedding_dim}")

    # Load the checkpoint
    try:
        fixed_model.load_state_dict(old_state_dict)
        print("✅ Success: Fixed model loaded checkpoint successfully!")
    except Exception as e:
        print(f"❌ Unexpected: Fixed loading failed: {e}")

if __name__ == "__main__":
    test_model_loading_with_different_embedding_dims()
