#!/usr/bin/env python3
"""
Debug script to verify batch size and ttl_batches behavior.
Run this to test how ttl_batches changes with different batch sizes.
"""

import sys
import os
sys.path.append('src')

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train.training import Training
from unittest.mock import Mock, patch

def test_batch_size_behavior():
    """Test how ttl_batches changes with batch size."""
    print("=== Testing ttl_batches behavior with different batch sizes ===")

    # Create a mock dataset with 10023 samples (similar to your data)
    total_samples = 10023
    dummy_data = torch.randn(total_samples, 3, 224, 224)
    dummy_labels = torch.randint(0, 771, (total_samples,))  # 771 players
    dataset = TensorDataset(dummy_data, dummy_labels)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of classes (players): {len(set(dummy_labels.numpy()))}")
    print()

    # Test different batch sizes
    batch_sizes = [8, 16, 32, 64, 128]

    for batch_size in batch_sizes:
        # Create DataLoader with drop_last=True (same as training)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        ttl_batches = len(dataloader)
        used_samples = ttl_batches * batch_size

        print(f"Batch size: {batch_size:3d} | ttl_batches: {ttl_batches:4d} | Used samples: {used_samples:5d} | Dropped: {total_samples - used_samples:3d}")

    print()
    print("Expected behavior: ttl_batches should decrease as batch_size increases")

def test_training_class_batch_size():
    """Test that Training class correctly uses batch_size parameter."""
    print("\n=== Testing Training class batch_size parameter ===")

    # Mock the parameter registry
    with patch('src.config.parameter_registry.parameter_registry') as mock_registry:
        mock_registry.parameters = {
            'batch_size': Mock(config_path='training_config.batch_size'),
            'num_epochs': Mock(config_path='training_config.num_epochs'),
        }
        mock_registry.get_kwarg_or_config = Mock(side_effect=lambda param, kwargs_dict: kwargs_dict.get(param, 32))

        # Test with different batch sizes
        for test_batch_size in [8, 16, 32, 64]:
            training = Training(batch_size=test_batch_size, num_epochs=1)
            print(f"Requested batch_size: {test_batch_size} | Training.batch_size: {training.batch_size}")

            # Create mock dataset
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=10023)

            # Test DataLoader setup
            with patch('src.train.training.DataLoader') as mock_dataloader:
                training.setup_dataloader(mock_dataset, type='train')

                # Check what batch_size was passed to DataLoader
                call_args = mock_dataloader.call_args
                dataloader_batch_size = call_args[1]['batch_size']
                print(f"  -> DataLoader batch_size: {dataloader_batch_size}")

                # Calculate expected ttl_batches
                expected_ttl_batches = 10023 // dataloader_batch_size
                print(f"  -> Expected ttl_batches: {expected_ttl_batches}")
            print()

def simulate_google_cloud_environment():
    """Simulate Google Cloud environment with CUDA device."""
    print("\n=== Simulating Google Cloud Environment (CUDA) ===")

    # Force CUDA device simulation
    original_cuda_available = torch.cuda.is_available
    original_mps_available = torch.backends.mps.is_available

    # Mock CUDA as available, MPS as unavailable
    torch.cuda.is_available = lambda: True
    torch.backends.mps.is_available = lambda: False

    try:
        # Mock the parameter registry
        with patch('src.config.parameter_registry.parameter_registry') as mock_registry:
            mock_registry.parameters = {
                'batch_size': Mock(config_path='training_config.batch_size'),
                'num_epochs': Mock(config_path='training_config.num_epochs'),
            }
            mock_registry.get_kwarg_or_config = Mock(side_effect=lambda param, kwargs_dict: kwargs_dict.get(param, 32))

            print("Device detection:")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            print(f"  MPS available: {torch.backends.mps.is_available()}")

            # Test with different batch sizes
            for test_batch_size in [8, 16, 32, 64]:
                training = Training(batch_size=test_batch_size, num_epochs=1)
                print(f"\nRequested batch_size: {test_batch_size}")
                print(f"  Training.batch_size: {training.batch_size}")
                print(f"  Training.device: {training.device}")

                # Create mock dataset
                mock_dataset = Mock()
                mock_dataset.__len__ = Mock(return_value=10023)

                # Test DataLoader setup
                with patch('src.train.training.DataLoader') as mock_dataloader:
                    training.setup_dataloader(mock_dataset, type='train')

                    # Check DataLoader configuration
                    call_args = mock_dataloader.call_args
                    dataloader_kwargs = call_args[1]

                    print(f"  DataLoader batch_size: {dataloader_kwargs['batch_size']}")
                    print(f"  DataLoader pin_memory: {dataloader_kwargs['pin_memory']}")
                    print(f"  DataLoader num_workers: {dataloader_kwargs['num_workers']}")

                    # Calculate expected ttl_batches
                    expected_ttl_batches = 10023 // dataloader_kwargs['batch_size']
                    print(f"  Expected ttl_batches: {expected_ttl_batches}")

    finally:
        # Restore original functions
        torch.cuda.is_available = original_cuda_available
        torch.backends.mps.is_available = original_mps_available

if __name__ == "__main__":
    test_batch_size_behavior()
    test_training_class_batch_size()
    simulate_google_cloud_environment()
