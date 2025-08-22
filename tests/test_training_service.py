from types import SimpleNamespace

from src.services.training_service import _convert_request_to_kwargs


def test_convert_request_to_kwargs_includes_top_level_n_datasets():
    # Create a minimal request object with the expected attributes
    req = SimpleNamespace(
        tenant_id="tenant1",
        verbose=True,
        custom_name="run1",
        resume_from_checkpoint=True,
        wandb_tags=["tag1"],
        n_datasets_to_use=5,
        training_params=None,
        model_params=None,
    )

    kwargs = _convert_request_to_kwargs(req)

    assert "n_datasets_to_use" in kwargs
    assert kwargs["n_datasets_to_use"] == 5
    # No training_kwargs should be present when training_params is None
    assert "training_kwargs" not in kwargs
