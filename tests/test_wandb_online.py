import os
import uuid
import importlib.util
import tempfile
import pytest
import torch


def load_wandb_logger_module():
    path = os.path.join('src', 'train', 'wandb_logger.py')
    spec = importlib.util.spec_from_file_location('wandb_logger_mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.integration
def test_wandb_online_checkpoint_and_registry():
    """
    Integration test that runs against wandb online. This test will be skipped
    unless WANDB_API_KEY is present in the environment and the `wandb` package
    is installed. It performs a minimal init_run, saves a checkpoint and a
    model to the registry using unique names so it is safe to run in parallel.
    """
    if 'WANDB_API_KEY' not in os.environ:
        pytest.skip('WANDB_API_KEY not set; skipping online wandb integration test')

    try:
        import wandb  # noqa: F401
    except Exception:
        pytest.skip('wandb not installed; skipping online wandb integration test')

    mod = load_wandb_logger_module()

    # Create a logger instance
    logger = mod.WandbLogger(enabled=True)

    run_name = f"test_run_{uuid.uuid4().hex[:8]}"
    assert logger.init_run({'test': True}, run_name=run_name)

    # Small model
    model = torch.nn.Linear(8, 2)

    # Save a checkpoint
    ckpt_artifact = logger.save_checkpoint(
        epoch=1,
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        loss=0.0,
        model_name=f"test_model_{uuid.uuid4().hex[:6]}",
        model_config={'dummy': True}
    )

    assert ckpt_artifact is not None

    # Save the model to registry using a safely formatted collection name
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    # save_model_to_registry writes a local file named 'model.pth' by default
    try:
        logger.save_model_to_registry(model, collection_name=collection_name, alias='latest')
    finally:
        # Clean up local model file if present
        try:
            if os.path.exists('model.pth'):
                os.remove('model.pth')
        except Exception:
            pass

    # Finish the run
    logger.finish()
