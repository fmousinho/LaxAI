import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for this test")
def test_resume_from_cpu_checkpoint_to_gpu(monkeypatch):
    """Simulate a checkpoint saved on CPU and resume on GPU.

    Procedure:
    - Instantiate a Training object targeting CUDA
    - Create a separate model+optimizer on CPU and perform one step to populate optimizer state
    - Monkeypatch wandb_logger.load_checkpoint to return that CPU checkpoint dict
    - Call Training.check_for_checkpoint_resumption() which will invoke the resume path
    - Assert the training.model parameters and optimizer state tensors are on CUDA afterwards
    """
    try:
        from siamesenet import SiameseNet
        from training import Training
        from config.all_config import model_config, wandb_config
        from src.train import wandb_logger as wl_mod
        from wandb_logger import wandb_logger
    except Exception as e:
        pytest.skip(f"Could not import required modules: {e}")

    # Ensure wandb resume path is considered by Training.check_for_checkpoint_resumption
    monkeypatch.setattr(wandb_config, 'enabled', True)

    # Create a Training instance targeting CUDA
    device = torch.device('cuda')
    tr = Training(device=device)

    # Setup model+optimizer in training (this will move model to CUDA and create optimizer)
    try:
        tr.setup_model(SiameseNet, model_name="test_model_resume")
    except Exception as e:
        pytest.skip(f"Failed to setup Training.model: {e}")

    assert tr.optimizer is not None

    # Build a separate CPU model and optimizer to create a CPU-saved checkpoint dict
    try:
        cpu_model = SiameseNet()
    except Exception as e:
        pytest.skip(f"Could not instantiate SiameseNet for CPU checkpoint creation: {e}")

    cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)

    # Run a single forward/backward/step on CPU to populate optimizer.state tensors
    batch_size = 1
    channels = 3
    H = getattr(model_config, 'input_height', 120)
    W = getattr(model_config, 'input_width', 80)
    dummy = torch.randn(batch_size, channels, H, W)

    cpu_model.train()
    try:
        cpu_optimizer.zero_grad()
        emb = cpu_model(dummy)
        loss = emb.sum()
        loss.backward()
        cpu_optimizer.step()
    except Exception as e:
        pytest.skip(f"Failed to run a CPU training step to populate optimizer state: {e}")

    # Build checkpoint dict with CPU tensors
    checkpoint = {
        'epoch': 2,
        'model_state_dict': {k: v.cpu() for k, v in cpu_model.state_dict().items()},
        'optimizer_state_dict': cpu_optimizer.state_dict(),
        'loss': 0.123,
        'model_config': {}
    }

    # Monkeypatch wandb_logger.load_checkpoint to return our CPU checkpoint
    # Use the instance imported from wandb_logger
    monkeypatch.setattr(wandb_logger, 'load_checkpoint', lambda artifact_name, version='latest': checkpoint)

    # Now call the resume path on training which should load checkpoint then move model+optimizer to device
    start_epoch = tr.check_for_checkpoint_resumption()

    # Expect resumed epoch to be checkpoint['epoch'] + 1
    assert start_epoch == checkpoint['epoch'] + 1

    # Assert model parameters are on CUDA
    param_dev = next(tr.model.parameters()).device
    assert param_dev.type == 'cuda'

    # Collect all tensors from optimizer state and ensure they are on CUDA
    def _iter_tensors(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _iter_tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _iter_tensors(v)

    tensors = list(_iter_tensors(tr.optimizer.state))
    # There should be some tensors (exp_avg/exp_avg_sq) in Adam state after loading
    assert len(tensors) > 0, "No tensors found in optimizer.state after resume"
    for t in tensors:
        assert isinstance(t, torch.Tensor)
        assert t.device.type == 'cuda', f"Optimizer state tensor not on cuda: {t.device}"
