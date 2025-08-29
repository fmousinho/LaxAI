#!/usr/bin/env python3
"""
Production verification script for Google Cloud environment.
Run this on your Google Cloud instance to verify batch size behavior.
"""

import sys
import os
sys.path.append('src')

import torch
from torch.utils.data import DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

def verify_batch_size_behavior(dataset, batch_sizes_to_test=[8, 16, 32, 64, 128]):
    """Verify ttl_batches changes with batch size on actual dataset."""
    print("=== Verifying Batch Size Behavior on Google Cloud ===")
    print(f"Dataset type: {type(dataset).__name__}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()

    for batch_size in batch_sizes_to_test:
        # Create DataLoader with same settings as training
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # Same as training
            num_workers=0,   # Start with 0 to avoid issues
            pin_memory=torch.cuda.is_available()  # Same as training
        )

        ttl_batches = len(dataloader)
        used_samples = ttl_batches * batch_size
        dropped_samples = len(dataset) - used_samples

        print(f"Batch size: {batch_size:3d} | ttl_batches: {ttl_batches:4d} | Used: {used_samples:5d} | Dropped: {dropped_samples:3d}")

        # Test that we can actually iterate through the dataloader
        try:
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 3:  # Just test first few batches
                    break
            print(f"  ✓ Successfully iterated {batch_count} batches")
        except Exception as e:
            print(f"  ✗ Error iterating batches: {e}")

    print()

def test_training_class_integration(dataset):
    """Test Training class with actual dataset."""
    print("=== Testing Training Class Integration ===")

    from src.train.training import Training
    from unittest.mock import patch

    # Mock parameter registry
    with patch('src.config.parameter_registry.parameter_registry') as mock_registry:
        mock_registry.parameters = {
            'batch_size': type('MockParam', (), {'config_path': 'training_config.batch_size'})(),
            'num_epochs': type('MockParam', (), {'config_path': 'training_config.num_epochs'})(),
        }
        mock_registry.get_kwarg_or_config = lambda param, kwargs_dict: kwargs_dict.get(param, 32)

        for test_batch_size in [8, 16, 32]:
            print(f"\nTesting batch_size={test_batch_size}:")

            training = Training(batch_size=test_batch_size, num_epochs=1)
            print(f"  Training.batch_size: {training.batch_size}")
            print(f"  Training.device: {training.device}")

            # Setup dataloader
            training.setup_dataloader(dataset, type='train')

            # Calculate expected ttl_batches
            expected_ttl_batches = len(dataset) // test_batch_size
            actual_ttl_batches = len(training.dataloader)

            print(f"  Expected ttl_batches: {expected_ttl_batches}")
            print(f"  Actual ttl_batches: {actual_ttl_batches}")
            print(f"  Match: {'✓' if expected_ttl_batches == actual_ttl_batches else '✗'}")

def main():
    """Main verification function - call this with your actual dataset."""
    print("Google Cloud Batch Size Verification")
    print("=" * 50)

    # You'll need to load your actual dataset here
    # For example:
    # from src.train.dataset import LacrossePlayerDataset
    # from your_storage_client import get_storage_client
    #
    # storage_client = get_storage_client()
    # dataset = LacrossePlayerDataset(
    #     image_dir="your/dataset/path",
    #     storage_client=storage_client
    # )

    # Placeholder - replace with your actual dataset loading
    print("⚠️  Please modify this script to load your actual LacrossePlayerDataset")
    print("   Then uncomment the verification calls below")
    print()

    # Example of what you should do:
    print("To use this script:")
    print("1. Load your actual LacrossePlayerDataset")
    print("2. Uncomment the lines below:")
    print("   verify_batch_size_behavior(dataset)")
    print("   test_training_class_integration(dataset)")
    print()

    # Uncomment these after loading your dataset:
    # verify_batch_size_behavior(dataset)
    # test_training_class_integration(dataset)

if __name__ == "__main__":
    main()
