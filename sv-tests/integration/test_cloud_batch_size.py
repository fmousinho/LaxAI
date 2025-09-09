#!/usr/bin/env python3
"""
Test script to verify batch size flow with actual dataset on Google Cloud.
This script simulates the exact same flow as the API endpoint.
"""
import logging
import os
import sys
from typing import Any, Dict

# Add src to path
sys.path.insert(0, '/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/src')
sys.path.insert(0, '/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/services/service-api/src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_batch_size_with_actual_dataset():
    """Test batch size flow using the actual dataset and parameters."""
    import sys

    from config.parameter_registry import parameter_registry
    sys.path.insert(0, '../../services/service-api/src')
    from train_pipeline import TrainPipeline
    from training import Training
    from training_service import _convert_request_to_kwargs

    from shared_libs.common.google_storage import get_storage

    # Simulate the request that user is sending
    class MockRequest:
        def __init__(self):
            self.tenant_id = "tenant1"
            self.training_params = {
                "batch_size": 16,  # This should change ttl_batches
                "num_epochs": 10,
                "learning_rate": 0.001
            }
            self.model_params = {}
            self.eval_params = {}
            self.custom_name = None
            self.resume_from_checkpoint = True
            self.wandb_tags = []
            self.n_datasets_to_use = None

        def model_dump(self):
            return {
                "tenant_id": self.tenant_id,
                "training_params": self.training_params,
                "model_params": self.model_params,
                "eval_params": self.eval_params,
                "custom_name": self.custom_name,
                "resume_from_checkpoint": self.resume_from_checkpoint,
                "wandb_tags": self.wandb_tags,
                "n_datasets_to_use": self.n_datasets_to_use
            }

    request = MockRequest()

    print("=== Testing Batch Size Flow with Actual Dataset ===")
    print(f"Request batch_size: {request.training_params['batch_size']}")

    # Step 1: Convert request to kwargs (same as API)
    kwargs = _convert_request_to_kwargs(request)
    print(f"1. Converted kwargs training_kwargs: {kwargs.get('training_kwargs', {})}")

    # Step 2: Extract batch_size from kwargs
    training_kwargs = kwargs.get('training_kwargs', {})
    batch_size = training_kwargs.get('batch_size', 32)  # Default
    print(f"2. Extracted batch_size: {batch_size}")

    # Step 3: Create Training instance with these kwargs
    training = Training(**training_kwargs)
    print(f"3. Training.batch_size: {training.batch_size}")

    # Step 4: Create actual dataset (simulate TrainPipeline)
    try:
        storage_client = get_storage(request.tenant_id)

        # Find available datasets
        from common.google_storage import GCSPaths
        path_manager = GCSPaths()
        datasets_folder = path_manager.get_path("datasets_root")
        if datasets_folder:
            datasets = storage_client.list_blobs(prefix=datasets_folder, delimiter='/', exclude_prefix_in_return=True)
            datasets = list(datasets)
            print(f"4. Found datasets: {datasets[:3]}...")  # Show first 3

            if datasets:
                # Use first dataset for testing
                dataset_name = datasets[0].rstrip('/')
                print(f"5. Using dataset: {dataset_name}")

                # Get train folder path
                train_folder = path_manager.get_path("train_dataset", dataset_id=dataset_name)
                print(f"6. Train folder: {train_folder}")
                
                if train_folder is None:
                    print("❌ Could not get train folder path!")
                    return

                # Create actual dataset
                from config.all_config import training_config
                from config.transforms import get_transforms
                from dataset import LacrossePlayerDataset

                training_transforms = get_transforms('training')
                dataset = LacrossePlayerDataset(
                    image_dir=train_folder,
                    storage_client=storage_client,
                    transform=training_transforms,
                    min_images_per_player=training_config.min_images_per_player
                )

                print(f"7. Dataset created with {len(dataset)} images")

                # Step 5: Setup dataloader
                training.setup_dataloader(dataset, type='train')
                assert training.dataloader is not None, "DataLoader should be initialized after setup_dataloader"
                actual_ttl_batches = len(training.dataloader)
                print(f"8. Actual ttl_batches: {actual_ttl_batches}")

                # Step 6: Calculate expected
                expected_ttl_batches = len(dataset) // training.batch_size
                print(f"9. Expected ttl_batches: {expected_ttl_batches}")

                # Step 7: Check if they match
                match = actual_ttl_batches == expected_ttl_batches
                print(f"10. Match: {'✓' if match else '✗'}")

                if not match:
                    print("❌ ISSUE FOUND: ttl_batches doesn't match expected calculation!")
                    print(f"   Dataset size: {len(dataset)}")
                    print(f"   Batch size: {training.batch_size}")
                    print(f"   Expected: {expected_ttl_batches}")
                    print(f"   Actual: {actual_ttl_batches}")
                    print(f"   Difference: {abs(actual_ttl_batches - expected_ttl_batches)}")
                else:
                    print("✅ Batch size flow working correctly!")

                # Test with different batch size
                print("\n=== Testing with different batch size ===")
                test_batch_sizes = [8, 16, 32, 64]

                for test_bs in test_batch_sizes:
                    # Create new training instance with different batch size
                    test_kwargs = training_kwargs.copy()
                    test_kwargs['batch_size'] = test_bs

                    test_training = Training(**test_kwargs)
                    test_training.setup_dataloader(dataset, type='train')
                    assert test_training.dataloader is not None, "DataLoader should be initialized after setup_dataloader"
                    test_ttl_batches = len(test_training.dataloader)
                    expected_test = len(dataset) // test_bs

                    print(f"Batch size {test_bs:2d}: ttl_batches={test_ttl_batches:4d}, expected={expected_test:4d}, match={'✓' if test_ttl_batches == expected_test else '✗'}")

            else:
                print("❌ No datasets found!")
        else:
            print("❌ Could not get datasets_root path!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_size_with_actual_dataset()
