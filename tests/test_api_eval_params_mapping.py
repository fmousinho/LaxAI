import sys
from types import SimpleNamespace


def test_convert_request_to_kwargs_maps_eval_params():
    """Ensure the service converts request.eval_params into eval_kwargs in the kwargs dict."""
    sys.path.append('src')
    from src.services.training_service import _convert_request_to_kwargs

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
