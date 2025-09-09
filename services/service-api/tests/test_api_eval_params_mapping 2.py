import sys
from types import SimpleNamespace


def test_convert_request_to_kwargs_maps_eval_params():
    """Ensure the service converts request.eval_params into eval_kwargs in the kwargs dict."""
    sys.path.append('src')
    from training_service import _convert_request_to_kwargs

    req = SimpleNamespace(
        tenant_id='tenant1',
        verbose=True,
        custom_name='run1',
        resume_from_checkpoint=True,
        wandb_tags=['tag1'],
        n_datasets_to_use=None,
        training_params=None,
        model_params=None,
        eval_params={'batch_size': 4, 'num_workers': 0},
    )

    kwargs = _convert_request_to_kwargs(req)

    assert 'eval_kwargs' in kwargs, 'eval_kwargs should be present in converted kwargs'
    assert kwargs['eval_kwargs'] == {'batch_size': 4, 'num_workers': 0}


def test_convert_request_to_kwargs_maps_training_params():
    """Ensure the service converts request.training_params into training_kwargs in the kwargs dict."""
    sys.path.append('src')
    from training_service import _convert_request_to_kwargs

    req = SimpleNamespace(
        tenant_id='tenant1',
        verbose=True,
        custom_name='run1',
        resume_from_checkpoint=True,
        wandb_tags=['tag1'],
        n_datasets_to_use=None,
        training_params={'batch_size': 8, 'num_epochs': 10, 'learning_rate': 0.001},
        model_params=None,
        eval_params=None,
    )

    kwargs = _convert_request_to_kwargs(req)

    assert 'training_kwargs' in kwargs, 'training_kwargs should be present in converted kwargs'
    assert kwargs['training_kwargs'] == {'batch_size': 8, 'num_epochs': 10, 'learning_rate': 0.001}


def test_training_batch_size_flow_from_api_to_dataloader():
    """Test that batch_size from API flows correctly to DataLoader configuration."""
    import sys
    sys.path.append('src')
    from training_service import _convert_request_to_kwargs
    from src.train.training import Training
    from unittest.mock import Mock, patch
    import torch

    # Mock the dataset and other dependencies
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)

    # Create a request with custom batch_size
    req = SimpleNamespace(
        tenant_id='tenant1',
        verbose=True,
        custom_name='test_run',
        resume_from_checkpoint=False,
        wandb_tags=[],
        n_datasets_to_use=None,
        training_params={'batch_size': 4, 'num_epochs': 1},
        model_params=None,
        eval_params=None,
    )

    # Convert request to kwargs
    kwargs = _convert_request_to_kwargs(req)

    # Verify training_kwargs contains our batch_size
    assert 'training_kwargs' in kwargs
    assert kwargs['training_kwargs']['batch_size'] == 4

    # Mock the parameter registry to avoid config dependencies
    with patch('src.config.parameter_registry.parameter_registry') as mock_registry:
        # Mock the parameters dict
        mock_registry.parameters = {
            'batch_size': Mock(config_path='training_config.batch_size'),
            'num_epochs': Mock(config_path='training_config.num_epochs'),
        }
        mock_registry.get_kwarg_or_config = Mock(side_effect=lambda param, kwargs_dict: kwargs_dict.get(param, 1))

        # Create Training instance with our kwargs
        training = Training(**kwargs['training_kwargs'])

        # Verify batch_size was set correctly
        assert training.batch_size == 4, f"Expected batch_size=4, got {training.batch_size}"

        # Test dataloader setup with our custom batch_size
        with patch('src.train.training.DataLoader') as mock_dataloader:
            training.setup_dataloader(mock_dataset, type='train')

            # Verify DataLoader was called with correct batch_size
            mock_dataloader.assert_called_once()
            call_args = mock_dataloader.call_args
            assert call_args[1]['batch_size'] == 4, f"DataLoader should use batch_size=4, got {call_args[1]['batch_size']}"


def test_end_to_end_batch_size_verification():
    """Integration test showing complete batch_size flow from API to DataLoader."""
    import sys
    sys.path.append('src')
    from training_service import _convert_request_to_kwargs
    from src.train.training import Training
    from unittest.mock import Mock, patch
    import torch

    print("\n=== END-TO-END BATCH SIZE VERIFICATION ===")

    # Step 1: Simulate API request with batch_size=8
    api_request = {
        'tenant_id': 'test-tenant',
        'verbose': True,
        'custom_name': 'batch_size_test',
        'training_params': {'batch_size': 8, 'num_epochs': 2, 'learning_rate': 0.001},
        'model_params': None,
        'eval_params': None
    }

    print(f"1. API Request batch_size: {api_request['training_params']['batch_size']}")

    # Step 2: Convert request to kwargs (service layer)
    kwargs = _convert_request_to_kwargs(SimpleNamespace(**api_request))
    print(f"2. Service Layer training_kwargs batch_size: {kwargs['training_kwargs']['batch_size']}")

    # Step 3: Initialize Training class (training layer)
    with patch('src.config.parameter_registry.parameter_registry') as mock_registry:
        mock_registry.parameters = {
            'batch_size': Mock(config_path='training_config.batch_size'),
            'num_epochs': Mock(config_path='training_config.num_epochs'),
            'learning_rate': Mock(config_path='training_config.learning_rate'),
        }
        mock_registry.get_kwarg_or_config = Mock(side_effect=lambda param, kwargs_dict: kwargs_dict.get(param, 1))

        training = Training(**kwargs['training_kwargs'])
        print(f"3. Training Class batch_size attribute: {training.batch_size}")

        # Step 4: Setup DataLoader (DataLoader layer)
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=160)  # 160 samples

        with patch('src.train.training.DataLoader') as mock_dataloader:
            training.setup_dataloader(mock_dataset, type='train')

            # Verify DataLoader configuration
            call_args = mock_dataloader.call_args
            dataloader_batch_size = call_args[1]['batch_size']
            print(f"4. DataLoader batch_size parameter: {dataloader_batch_size}")

            # Calculate expected number of batches
            expected_batches = 160 // dataloader_batch_size  # Should be 160/8 = 20
            print(f"5. Expected number of batches: {expected_batches}")

            # Verify all values match
            assert api_request['training_params']['batch_size'] == 8
            assert kwargs['training_kwargs']['batch_size'] == 8
            assert training.batch_size == 8
            assert dataloader_batch_size == 8

            print("✅ SUCCESS: Batch size flows correctly through all layers!")
            print("   API (8) → Service (8) → Training Class (8) → DataLoader (8)")
