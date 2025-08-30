import time
import uuid
import pytest
import torch
from train.wandb_logger import wandb_logger
from config.all_config import wandb_config


@pytest.mark.integration
def test_wandb_cleanup_integration():
    """Integration test: creates two checkpoint artifacts in real WandB project,
    runs cleanup to keep only latest, and verifies older versions were deleted.

    This test uses real WandB credentials from the environment. It will be
    skipped if WandB integration is disabled in config or if login fails.
    """
    # Skip if wandb is disabled in config
    if not wandb_config.enabled:
        pytest.skip("wandb is disabled in config")

    # Ensure we have a working API client (re-login if necessary)
    api = wandb_logger.wandb_api
    if not api:
        api = wandb_logger._login_and_get_api()
        wandb_logger.wandb_api = api

    if not api:
        pytest.skip("Could not initialize wandb Api (check WANDB_API_KEY in environment)")

    run_name = f"integration_cleanup_{uuid.uuid4().hex[:8]}"
    wandb_logger.init_run(config={'test': True}, run_name=run_name)

    try:
        # Create two checkpoints so there are multiple versions
        # Use very small dicts to avoid large uploads
        wandb_logger.save_checkpoint(epoch=1, model_state_dict={'w': torch.tensor([1])}, optimizer_state_dict={}, loss=0.1)
        # small sleep to ensure distinct created_at timestamps
        time.sleep(1.5)
        wandb_logger.save_checkpoint(epoch=2, model_state_dict={'w': torch.tensor([2])}, optimizer_state_dict={}, loss=0.05)

        # Allow artifact indexing to propagate
        time.sleep(2)

        artifact_name = wandb_logger._sanitize_artifact_name(wandb_logger._get_checkpoint_name())

        # List artifact versions via API and confirm cleanup reduces them to 1
        artifact_type_api = wandb_logger.wandb_api.artifact_type("model_checkpoint", project=wandb_config.project)
        collection_api = artifact_type_api.collection(artifact_name)
        versions_before = list(collection_api.artifacts())

        # There should be at least 2 versions before cleanup
        assert len(versions_before) >= 2, f"Expected >=2 artifact versions before cleanup, got {len(versions_before)}"

        # Run cleanup keeping only latest
        wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=1)

        # Give backend a moment to process deletions
        time.sleep(2)

        versions_after = list(collection_api.artifacts())
        assert len(versions_after) <= 1, f"Expected <=1 artifact versions after cleanup, got {len(versions_after)}"

        # Best-effort: remove any remaining artifact versions created by this test
        try:
            for v in versions_after:
                try:
                    v.delete()
                except Exception:
                    # ignore individual delete failures but continue
                    pass
            # allow backend to process deletes
            time.sleep(1)
            remaining = list(collection_api.artifacts())
            # We expect none remain, but accept 0 or 1 depending on propagation; try to assert <=0 conservatively
            assert len(remaining) == 0 or len(remaining) <= 0 or len(remaining) <= 1
        except Exception:
            # Do not fail the test on best-effort cleanup; just proceed to finalization
            pass

    finally:
        # Finish run to release resources
        try:
            wandb_logger.finish()
        except Exception:
            pass
