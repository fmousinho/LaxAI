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
    from utils.env_secrets import setup_environment_secrets
    setup_environment_secrets()
except Exception:
    # Re-raise so failures are visible during collection. Tests that run in
    # environments without these secrets should set them before invoking pytest.
    raise
