import os
import toml
from dotenv import load_dotenv
import utils.env_secrets as env_secrets


def test_setup_environment_secrets_runs_and_reports():
    """Run setup_environment_secrets() and fail the test with the exception text if it raises."""
    repo_root = os.path.abspath('.')
    load_dotenv(os.path.join(repo_root, '.env'))
    env_secrets.CONFIG = toml.load(os.path.join(repo_root, 'config.toml'))

    try:
        env_secrets.setup_environment_secrets()
    except Exception as e:
        raise AssertionError(f'setup_environment_secrets() raised: {type(e).__name__}: {e}')
    

    # After running setup, assert that all secrets listed in config.toml
    # are present in the environment and non-empty.
    cfg = toml.load(os.path.join(repo_root, 'config.toml'))
    secrets = cfg.get('secrets', [])
    if isinstance(secrets, dict) and 'secrets' in secrets:
        secrets = secrets['secrets']
    if isinstance(secrets, str):
        secrets = [secrets]

    if len(secrets) == 0:
        raise AssertionError('No secrets listed in config.toml to verify after setup_environment_secrets()')

    missing = [s for s in secrets if not os.environ.get(s)]
    if missing:
        raise AssertionError(f'Missing secrets in environment after setup: {missing}')
