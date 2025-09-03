import sys
import os

# Ensure the repository root and the src/ directory are on sys.path so tests
# can import application modules (scripts, utils, src.*) regardless of how
# pytest is invoked.
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ensure environment secrets are loaded before any tests import modules that
# depend on them. This will raise on missing required secrets (fail-fast),
# matching application behavior. If you prefer tests to run without secrets,
# adjust this call or set the environment variables in your CI/test runner.
try:
    # Import here so tests get the repo's `src` on sys.path first
    # Do NOT inject placeholder WANDB/HUGGINGFACE tokens here. The
    # application expects those secrets to be provided via
    # `setup_environment_secrets()` (env/.env/Secret Manager). Tests
    # should rely on that mechanism so behavior matches production.

    # Ensure GOOGLE_CLOUD_PROJECT is present for Secret Manager checks. Prefer
    # an existing environment value, then a value from a .env file if present.
    if 'GOOGLE_CLOUD_PROJECT' not in os.environ:
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        env_path = os.path.normpath(env_path)
        try:
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            k, v = line.split('=', 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k == 'GOOGLE_CLOUD_PROJECT' and v:
                                os.environ['GOOGLE_CLOUD_PROJECT'] = v
                                break
        except Exception:
            # Best-effort: if reading .env fails, continue and let setup fail
            pass

    # Import here so tests get the repo's `src` on sys.path first
    from src.utils.env_secrets import setup_environment_secrets
    setup_environment_secrets()
except Exception:
    # Re-raise so failures are visible during collection. Tests that run in
    # environments without these secrets should set them before invoking pytest.
    raise


# WandB test cleanup utilities
import pytest
import time
from typing import List


class WandbArtifactCleaner:
    """Manages cleanup of WandB artifacts created during tests."""
    
    def __init__(self):
        self.artifacts_to_cleanup: List[str] = []
        
    def track_artifact(self, artifact_name: str):
        """Track an artifact name for cleanup after test."""
        if artifact_name not in self.artifacts_to_cleanup:
            self.artifacts_to_cleanup.append(artifact_name)
            
    def cleanup_all(self):
        """Clean up all tracked artifacts."""
        if not self.artifacts_to_cleanup:
            return
            
        try:
            from src.train.wandb_logger import wandb_logger
            for artifact_name in self.artifacts_to_cleanup:
                try:
                    wandb_logger._cleanup_old_checkpoints(artifact_name, keep_latest=0)
                    time.sleep(0.2)  # Brief pause for propagation
                except Exception as e:
                    print(f"Warning: Failed to cleanup artifact {artifact_name}: {e}")
        except ImportError:
            # If wandb_logger not available, skip cleanup
            pass
        finally:
            self.artifacts_to_cleanup.clear()


@pytest.fixture
def wandb_artifact_cleaner():
    """Pytest fixture for automatic WandB artifact cleanup."""
    cleaner = WandbArtifactCleaner()
    yield cleaner
    # Cleanup happens after test completes (success or failure)
    cleaner.cleanup_all()
