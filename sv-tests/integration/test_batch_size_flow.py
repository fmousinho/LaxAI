#!/usr/bin/env python3
"""
Test to verify batch size flow from API to DataLoader in the actual pipeline.
This simulates the exact same flow as your Google Cloud setup.
"""

import os
import sys

sys.path.append('src')

def test_complete_batch_size_flow():
    """Test the complete batch size flow from API request to DataLoader."""
    print("=== Complete Batch Size Flow Test ===")

    # Step 1: Simulate API request with batch_size
    from types import SimpleNamespace

    api_request = SimpleNamespace(
        tenant_id='test-tenant',
        verbose=True,
        custom_name='batch_size_test',
        resume_from_checkpoint=False,
        wandb_tags=[],
        n_datasets_to_use=None,
        training_params={'batch_size': 16, 'num_epochs': 1, 'learning_rate': 0.001},
        model_params=None,
        eval_params=None,
    )

    print(f"1. API Request batch_size: {api_request.training_params['batch_size']}")

    # Step 2: Convert request to kwargs (service layer)
    import sys
    sys.path.insert(0, '../../services/service-api/src')
    from training_service import _convert_request_to_kwargs
    kwargs = _convert_request_to_kwargs(api_request)
    print(f"2. Service Layer training_kwargs batch_size: {kwargs['training_kwargs']['batch_size']}")

    # Step 3: Initialize Training class
    from unittest.mock import patch

    from training_loop import Training

    with patch('services.service_training.src.config.parameter_registry.parameter_registry') as mock_registry:
        mock_registry.parameters = {
            'batch_size': type('MockParam', (), {'config_path': 'training_config.batch_size'})(),
            'num_epochs': type('MockParam', (), {'config_path': 'training_config.num_epochs'})(),
            'learning_rate': type('MockParam', (), {'config_path': 'training_config.learning_rate'})(),
        }
        mock_registry.get_kwarg_or_config = lambda param, kwargs_dict: kwargs_dict.get(param, 32)

        training = Training(**kwargs['training_kwargs'])
        print(f"3. Training Class batch_size attribute: {training.batch_size}")

        # Step 4: Create mock dataset
        import torch
        from torch.utils.data import TensorDataset

        total_samples = 10023
        dummy_data = torch.randn(total_samples, 3, 224, 224)
        dummy_labels = torch.randint(0, 771, (total_samples,))
        dataset = TensorDataset(dummy_data, dummy_labels)

        print(f"4. Dataset size: {len(dataset)} samples")

        # Step 5: Setup DataLoader
        training.setup_dataloader(dataset, type='train')

        # Step 6: Verify DataLoader configuration
        assert training.dataloader is not None, "DataLoader should be initialized after setup_dataloader"
        actual_ttl_batches = len(training.dataloader)
        expected_ttl_batches = len(dataset) // training.batch_size

        print(f"5. DataLoader batch_size: {training.dataloader.batch_size}")
        print(f"6. DataLoader drop_last: {training.dataloader.drop_last}")
        print(f"7. Actual ttl_batches: {actual_ttl_batches}")
        print(f"8. Expected ttl_batches: {expected_ttl_batches}")
        print(f"9. Match: {'✓' if actual_ttl_batches == expected_ttl_batches else '✗'}")

        # Step 7: Test with different batch size
        print("\n--- Testing with different batch size ---")
        api_request.training_params['batch_size'] = 32
        kwargs = _convert_request_to_kwargs(api_request)
        training2 = Training(**kwargs['training_kwargs'])

        print(f"New API batch_size: {api_request.training_params['batch_size']}")
        print(f"New Training batch_size: {training2.batch_size}")

        training2.setup_dataloader(dataset, type='train')
        assert training2.dataloader is not None, "DataLoader should be initialized after setup_dataloader"
        new_ttl_batches = len(training2.dataloader)
        new_expected = len(dataset) // training2.batch_size

        print(f"New ttl_batches: {new_ttl_batches}")
        print(f"New expected: {new_expected}")
        print(f"New match: {'✓' if new_ttl_batches == new_expected else '✗'}")

        # Verify they are different
        print(f"Batch sizes different: {'✓' if training.batch_size != training2.batch_size else '✗'}")
        print(f"ttl_batches different: {'✓' if actual_ttl_batches != new_ttl_batches else '✗'}")

if __name__ == "__main__":
    test_complete_batch_size_flow()
