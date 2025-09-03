import torch
import torch.nn as nn
import types

from src.train import wandb_logger


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # lazy head
        self._head = None

    def forward(self, x):
        # If head isn't initialized, create a trivial mapping
        if self._head is None:
            # simulate lazy creation on first forward
            self._head = nn.Sequential(nn.Dropout(0.0), nn.Linear(x.shape[-1], 2))
        out = x.mean(dim=[-2, -1]) if x.ndim == 4 else x
        if hasattr(self, '_head') and self._head is not None:
            return self._head(out)
        return out


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DummyBackbone()

    def forward(self, x):
        return self.backbone(x)

    def ensure_head_initialized(self, device: str = 'cpu', input_shape=(1, 3, 4)) -> None:
        # create the head with expected input features
        dev = torch.device(device)
        # Use a small dummy linear matching flattened input dim
        feat_dim = input_shape[-1]
        if getattr(self.backbone, '_head', None) is None:
            self.backbone._head = nn.Sequential(nn.Dropout(0.0), nn.Linear(feat_dim, 2))
            self.backbone._head.to(dev)


def test_resume_handles_lazy_head_and_loads_weights(tmp_path, monkeypatch):
    # Build a model that already has a head so we can export a matching state dict
    model_with_head = DummyModel()
    model_with_head.ensure_head_initialized(device='cpu', input_shape=(1, 3, 4))

    # Initialize head weights deterministically
    for p in model_with_head.backbone._head.parameters():
        nn.init.constant_(p, 0.5)

    state = model_with_head.state_dict()

    # Craft checkpoint dict similar to what save_checkpoint produces
    checkpoint = {
        'epoch': 2,
        'model_state_dict': state,
        # intentionally omit optimizer_state_dict to avoid needing exact optimizer
    }

    # Patch the WandbLogger instance's load_checkpoint to return our checkpoint
    monkeypatch.setattr(wandb_logger.wandb_logger, 'load_checkpoint', lambda name, version='latest': checkpoint)

    # Create a fresh model (without head) and optimizer
    model_to_resume = DummyModel()
    # Ensure backbone._head is None initially
    assert getattr(model_to_resume.backbone, '_head', None) is None

    # Ensure optimizer has at least one parameter: attach a small dummy parameter
    model_to_resume._dummy_param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD(model_to_resume.parameters(), lr=0.01)

    # Run resume - this should trigger ensure_head_initialized and successful load
    start_epoch = wandb_logger.wandb_logger.resume_training_from_checkpoint(model_to_resume, opt, artifact_name='dummy')

    assert start_epoch == checkpoint['epoch'] + 1

    # After resume, head should be initialized and weights should match
    assert getattr(model_to_resume.backbone, '_head', None) is not None
    resumed_state = model_to_resume.state_dict()

    # Compare one parameter tensor from the head to ensure it was restored
    key = [k for k in state.keys() if k.startswith('backbone._head')][0]
    assert torch.allclose(state[key], resumed_state[key])
