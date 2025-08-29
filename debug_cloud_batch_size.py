#!/usr/bin/env python3
"""
Debug script to run on Google Cloud to diagnose ttl_batches issue.
This script checks dataset size, batch size flow, and logs detailed information.
"""
import os
import sys
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_cloud_batch_size():
    """Debug batch size flow on Google Cloud."""
    print("=== Google Cloud Batch Size Debug ===")

    try:
        # Import required modules
        from config.parameter_registry import parameter_registry
        from services.training_service import _convert_request_to_kwargs
        from train.training import Training
        from common.google_storage import get_storage
        from config.all_config import training_config

        # Test with the same parameters the user is using
        test_cases = [
            {"batch_size": 16, "expected_images": 10023, "expected_players": 771},
            {"batch_size": 32, "expected_images": 10023, "expected_players": 771},
            {"batch_size": 64, "expected_images": 10023, "expected_players": 771},
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1}: batch_size={test_case['batch_size']} ---")

            # Simulate API request
            class MockRequest:
                def __init__(self, batch_size):
                    self.tenant_id = "tenant1"  # Adjust if different
                    self.training_params = {
                        "batch_size": batch_size,
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

            request = MockRequest(test_case['batch_size'])

            # Step 1: Convert request (same as API)
            kwargs = _convert_request_to_kwargs(request)
            training_kwargs = kwargs.get('training_kwargs', {})
            logger.info(f"Converted training_kwargs: {training_kwargs}")

            # Step 2: Create Training instance
            training = Training(**training_kwargs)
            logger.info(f"Training.batch_size: {training.batch_size}")

            # Step 3: Get storage client
            storage_client = get_storage(request.tenant_id)
            logger.info("Storage client created successfully")

            # Step 4: Find datasets
            from common.google_storage import GCSPaths
            path_manager = GCSPaths()
            datasets_folder = path_manager.get_path("datasets_root")

            if not datasets_folder:
                logger.error("Could not get datasets_root path!")
                continue

            datasets = storage_client.list_blobs(prefix=datasets_folder, delimiter='/', exclude_prefix_in_return=True)
            datasets = list(datasets)
            logger.info(f"Found {len(datasets)} datasets: {datasets[:3]}...")

            if not datasets:
                logger.error("No datasets found!")
                continue

            # Step 5: Use first dataset
            dataset_name = datasets[0].rstrip('/')
            logger.info(f"Using dataset: {dataset_name}")

            # Step 6: Get train folder
            train_folder = path_manager.get_path("train_dataset", dataset_id=dataset_name)
            logger.info(f"Train folder: {train_folder}")

            # Step 7: Create dataset
            from train.dataset import LacrossePlayerDataset
            from config.transforms import get_transforms

            training_transforms = get_transforms('training')
            dataset = LacrossePlayerDataset(
                image_dir=train_folder,
                storage_client=storage_client,
                transform=training_transforms,
                min_images_per_player=training_config.min_images_per_player
            )

            actual_images = len(dataset)
            logger.info(f"Dataset created with {actual_images} images")
            logger.info(f"Expected images: {test_case['expected_images']}")

            # Check if dataset size matches expectation
            if actual_images != test_case['expected_images']:
                logger.warning(f"❌ Dataset size mismatch! Expected: {test_case['expected_images']}, Actual: {actual_images}")

            # Step 8: Setup dataloader
            training.setup_dataloader(dataset, type='train')
            actual_ttl_batches = len(training.dataloader)
            expected_ttl_batches = actual_images // training.batch_size

            logger.info(f"Batch size: {training.batch_size}")
            logger.info(f"Actual ttl_batches: {actual_ttl_batches}")
            logger.info(f"Expected ttl_batches: {expected_ttl_batches}")
            logger.info(f"Match: {'✓' if actual_ttl_batches == expected_ttl_batches else '✗'}")

            if actual_ttl_batches != expected_ttl_batches:
                logger.error("❌ ISSUE FOUND: ttl_batches calculation is wrong!")
                logger.error(f"   Formula: floor({actual_images} / {training.batch_size}) = {expected_ttl_batches}")
                logger.error(f"   Actual result: {actual_ttl_batches}")
            else:
                logger.info("✅ ttl_batches calculation is correct")

            # Log detailed dataloader info
            logger.info("DataLoader details:")
            logger.info(f"  - num_workers: {training.dataloader.num_workers}")
            logger.info(f"  - pin_memory: {training.dataloader.pin_memory}")
            logger.info(f"  - drop_last: True (for train)")
            logger.info(f"  - shuffle: True (for train)")

        print("\n=== Environment Information ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Training config min_images_per_player: {training_config.min_images_per_player}")

        # Check if there are any global variables that might be persisting
        print("\n=== Checking for Global State ===")

        # Check parameter registry
        logger.info(f"Parameter registry has {len(parameter_registry.parameters)} parameters")

        # Check if dataset has any cached state
        if 'dataset' in locals():
            logger.info(f"Dataset players: {len(dataset.players) if hasattr(dataset, 'players') else 'N/A'}")
            logger.info(f"Dataset all_images length: {len(dataset.all_images) if hasattr(dataset, 'all_images') else 'N/A'}")

    except Exception as e:
        logger.error(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_cloud_batch_size()
