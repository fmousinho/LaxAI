import importlib.util
import os
import uuid

import pytest
import torch

# Note: All WandB tests should use max 2 epochs and single dataset unless specifically required
# See tests/test_config.py for test performance constraints


def load_wandb_logger_module():
    path = os.path.join('services', 'service_training', 'src', 'wandb_logger.py')
    spec = importlib.util.spec_from_file_location('wandb_logger_mod', path)
    if spec is None:
        raise ImportError(f"Could not load spec for {path}")
    if spec.loader is None:
        raise ImportError(f"Could not load loader for {path}")
    # Type assertions to help Pylance understand these are not None
    assert spec is not None
    assert spec.loader is not None
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

    # Clean up WandB artifacts created during this test
    try:
        # Clean up checkpoint artifacts
        if ckpt_artifact:
            checkpoint_name = logger._sanitize_artifact_name(logger._get_checkpoint_name())
            logger._cleanup_old_checkpoints(checkpoint_name, keep_latest=0)  # Remove all checkpoint versions
        
        # Clean up model artifacts
        logger._cleanup_old_model_versions(collection_name, keep_latest=0)  # Remove all model versions
        
        logger.info(f"Cleaned up test artifacts: checkpoint={checkpoint_name}, model={collection_name}")
    except Exception as e:
        logger.warning(f"Failed to cleanup test artifacts: {e}")

    # Finish the run
    logger.finish()


@pytest.fixture(scope="session", autouse=True)
def cleanup_wandb_online_artifacts():
    """Comprehensive cleanup of all test artifacts after test session."""
    yield  # Run tests first
    
    # Comprehensive cleanup after all tests complete
    try:
        from config.all_config import wandb_config

        import wandb
        
        api = wandb.Api()
        
        # Clean up all test artifacts that start with "test-" or contain test identifiers
        try:
            # Get all artifact types
            artifact_types = api.artifact_types(project=f"{wandb_config.team}/{wandb_config.project}")

            total_deleted = 0
            for artifact_type in artifact_types:
                try:
                    # Get all collections for this artifact type
                    collections = artifact_type.collections()

                    for collection in collections:
                        try:
                            # Check if collection name starts with "test-" or contains test identifiers
                            if (collection.name.startswith("test-") or 
                                "test_collection" in collection.name or
                                "test_run" in collection.name or
                                "test_model" in collection.name):
                                artifacts = list(collection.artifacts())

                                for artifact in artifacts:
                                    try:
                                        # Try to delete the artifact directly
                                        artifact.delete()
                                        print(f"🧹 Cleaned up test artifact: {artifact.name}")
                                        total_deleted += 1
                                    except Exception as e:
                                        error_msg = str(e).lower()
                                        # If deletion failed due to alias, try to remove alias first
                                        if "alias" in error_msg or "409" in str(e):
                                            try:
                                                print(f"⚠️ Artifact {artifact.name} has alias, attempting to remove alias and retry...")
                                                # Try to delete by removing aliases first
                                                if hasattr(artifact, 'aliases') and artifact.aliases:
                                                    try:
                                                        # Try to remove aliases
                                                        for alias in artifact.aliases[:]:  # Copy the list to avoid modification issues
                                                            try:
                                                                # Remove alias
                                                                if alias in artifact.aliases:
                                                                    artifact.aliases.remove(alias)
                                                                    artifact.save()
                                                                print(f"ℹ️ Removed alias '{alias}' from {artifact.name}")
                                                            except Exception as alias_error:
                                                                print(f"⚠️ Failed to remove alias '{alias}' from {artifact.name}: {alias_error}")
                                                    except Exception as collection_error:
                                                        print(f"⚠️ Failed to access collection for {artifact.name}: {collection_error}")

                                                # Try to delete again after removing aliases
                                                artifact.delete()
                                                print(f"🧹 Cleaned up test artifact (after alias removal): {artifact.name}")
                                                total_deleted += 1
                                            except Exception as retry_error:
                                                print(f"⚠️ Failed to delete test artifact {artifact.name} even after alias removal: {retry_error}")
                                        else:
                                            print(f"⚠️ Failed to delete test artifact {artifact.name}: {e}")

                        except Exception as e:
                            print(f"⚠️ Failed to access collection {collection.name}: {e}")

                except Exception as e:
                    print(f"⚠️ Failed to access artifact type {artifact_type.name}: {e}")

            if total_deleted > 0:
                print(f"🧹 Cleaned up {total_deleted} test artifacts total from online test suite")
            else:
                print("ℹ️ No test artifacts found to clean up")

        except Exception as e:
            print(f"⚠️ Failed to clean up test artifacts: {e}")

    except Exception as e:
        print(f"⚠️ Failed to initialize wandb API for cleanup: {e}")
