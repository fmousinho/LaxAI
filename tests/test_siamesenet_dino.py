import os
import toml
import pytest

import utils.env_secrets as env_secrets


def test_siamesenet_dino_can_download_and_initialize(tmp_path):
    """Integration test: ensure SiameseNet downloads DINOv3 from Hugging Face and initializes.

    This test will call `setup_environment_secrets()` to ensure credentials are present
    (it may read from env/.env or Google Secret Manager). It will fail if the download
    or model initialization fails.
    """

    # Ensure environment secrets are set for the test run
    env_secrets.setup_environment_secrets()

    # Ensure the Hugging Face token is available to the process; the model is gated
    # and will fail to download without a valid HUGGINGFACE_HUB_TOKEN.
    assert 'HUGGINGFACE_HUB_TOKEN' in os.environ and os.environ.get('HUGGINGFACE_HUB_TOKEN'), \
        "HUGGINGFACE_HUB_TOKEN not found in environment after setup_environment_secrets()"

    # Optionally set HF cache to a temporary directory to avoid polluting user cache
    os.environ.setdefault('HF_HOME', str(tmp_path / 'hf_cache'))

    try:
        # Import after setting up credentials to ensure HF token is available
        from src.train.siamesenet_dino import SiameseNet

        # Instantiate; defaults use the configured model_config dino_variant
        net = SiameseNet()

        # Basic sanity: forward a small random tensor to ensure model is usable
        import torch

        net.eval()
        with torch.no_grad():
            # create a single dummy image tensor with channels=3 and default size
            h, w = getattr(net.backbone, '_expected_input_size', (224, 224))
            x = torch.randn(1, 3, h, w)
            emb = net(x)

        assert emb is not None

    except Exception as e:
        pytest.fail(f"SiameseNet failed to download/initialize: {type(e).__name__}: {e}")
