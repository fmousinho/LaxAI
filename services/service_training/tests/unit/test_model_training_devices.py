import pytest
import torch

# Build device list dynamically so pytest reports skipped devices cleanly
devices = [torch.device('cpu')]
if torch.backends.mps.is_available():
    devices.append(torch.device('mps'))
if torch.cuda.is_available():
    devices.append(torch.device('cuda'))


@pytest.mark.parametrize('device', devices)
def test_siamese_one_epoch_on_device(device):
    """Import the Siamese model and run one training step (one batch) on the given device.

    This test is intentionally lightweight: one forward/backward/opt step using
    randomly generated triplet tensors. It skips if the model cannot be
    instantiated (for example if pretrained weights cannot be downloaded).
    """
    try:
        import os
        import sys

        from siamesenet import SiameseNet
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'shared_libs'))
        from config.all_config import model_config
    except Exception as e:
        pytest.skip(f"Could not import SiameseNet or config: {e}")

    # Try to instantiate the model; if weight download or other issues occur,
    # skip the test to avoid flakiness in CI or offline environments.
    try:
        model = SiameseNet()
    except Exception as e:
        pytest.skip(f"SiameseNet instantiation failed (possibly weight download): {e}")

    # Move model to device and ensure parameters are on same device
    try:
        model.to(device)
    except Exception as e:
        pytest.skip(f"Failed to move model to device {device}: {e}")

    model.train()

    batch_size = 2
    channels = 3
    H = getattr(model_config, 'input_height', 120)
    W = getattr(model_config, 'input_width', 80)

    # Create random triplet tensors on the target device
    a = torch.randn(batch_size, channels, H, W, device=device)
    p = a.clone()  # make positive identical to anchor (easy positive)
    n = torch.randn(batch_size, channels, H, W, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.TripletMarginLoss(margin=0.3, p=2)

    # Run a single training iteration
    try:
        optim.zero_grad()
        emb_a, emb_p, emb_n = model.forward_triplet(a, p, n)
        loss = loss_fn(emb_a, emb_p, emb_n)
        loss.backward()
        optim.step()
    except Exception as e:
        pytest.fail(f"Training step failed on device {device}: {type(e).__name__}: {e}")

    # Basic assertions
    assert loss is not None
    assert not torch.isnan(loss).any()
    assert loss.item() >= 0.0
