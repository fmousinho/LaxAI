import os
import toml
from dotenv import load_dotenv
import utils.env_secrets as env_secrets


def test_setup_environment_secrets_picks_up_env_var():
    # Load module directly to avoid package __init__ side-effects
    mod = env_secrets

    # Ensure detection helpers do not perform network/colab checks
    mod.is_running_in_colab = lambda: False
    mod.is_running_in_gcp = lambda: False
    # Make sure the module uses the project's config.toml and .env in repo root
    repo_root = os.path.abspath('.')
    load_dotenv(os.path.join(repo_root, '.env'))
    mod.CONFIG = toml.load(os.path.join(repo_root, 'config.toml'))

    # Pick one of the secrets listed in config.toml and ensure it's in the env
    # Use WANDB_API_KEY which is present (commented out in .env, but present in config mapping)
    secret_env_name = 'WANDB_API_KEY'
    # Put the secret in the environment so the loader finds it from env first
    os.environ[secret_env_name] = 'supersecret123'

    # Call the function under test
    mod.setup_environment_secrets()

    # Assert the environment variable remains set with the expected value
    assert os.environ.get(secret_env_name) == 'supersecret123'


def test_setup_environment_secrets_raises_when_missing():
    mod = env_secrets
    mod.is_running_in_colab = lambda: False
    mod.is_running_in_gcp = lambda: False
    # Use a secret name that's not present anywhere
    missing_secret_name = 'THIS_SECRET_DOES_NOT_EXIST'
    # Ensure module uses project config
    repo_root = os.path.abspath('.')
    mod.CONFIG = toml.load(os.path.join(repo_root, 'config.toml'))

    # Replace secrets list with the missing one so load_secrets will raise
    mod.CONFIG['secrets'] = [{missing_secret_name: [missing_secret_name]}]

    # Ensure not present
    if missing_secret_name in os.environ:
        del os.environ[missing_secret_name]

    # Call load_secrets directly; it raises ValueError when secret isn't found
    try:
        mod.load_secrets(mod.CONFIG)
    except ValueError as e:
        assert missing_secret_name in str(e)
    else:
        raise AssertionError('Expected ValueError for missing secret but none was raised')


def test_secret_manager_retrieves_hf_token():
        """
        Integration test: ensure HUGGINGFACE_HUB_TOKEN can be retrieved from
        Google Secret Manager and set as an environment variable.

        This test does a real network call. It requires:
            - GOOGLE_CLOUD_PROJECT set in environment (or in .env)
            - Valid GCP credentials available (ADC or service account via file)
            - The secret `huggingface-hub-token` present in Secret Manager for the project
        """
        mod = env_secrets

        # Use project config from repo
        repo_root = os.path.abspath('.')
        mod.CONFIG = toml.load(os.path.join(repo_root, 'config.toml'))

        secret_var = 'HUGGINGFACE_HUB_TOKEN'

        # Ensure we don't accidentally read it from the environment
        if secret_var in os.environ:
                del os.environ[secret_var]

        # Call load_secrets which will try env -> .env -> Secret Manager
        mod.load_secrets(mod.CONFIG)

        # Expect the secret to now be present in the environment
        assert os.environ.get(secret_var), "HUGGINGFACE_HUB_TOKEN was not set from Secret Manager"


def test_huggingface_token_loaded_from_secret_manager():
    """HUGGINGFACE_HUB_TOKEN is only available in Secret Manager; simulate that path."""
    mod = env_secrets
    mod.is_running_in_colab = lambda: False
    mod.is_running_in_gcp = lambda: True

    # Ensure not present in environment or .env
    if 'HUGGINGFACE_HUB_TOKEN' in os.environ:
        del os.environ['HUGGINGFACE_HUB_TOKEN']

    # Configure module to report Secret Manager available and patch the getter
    mod._SECRETS_MANAGER_INSTALLED = True

    def fake_get_from_secret_manager(secret_name, project_id=None):
        if secret_name in ('huggingface-hub-token', 'HUGGINGFACE_HUB_TOKEN'):
            return 'hf_token_from_sm'
        return None

    mod._get_from_secret_manager = fake_get_from_secret_manager

    # Provide CONFIG expected by load_secrets: list of mappings
    mod.CONFIG = {'secrets': [{'HUGGINGFACE_HUB_TOKEN': ['huggingface-hub-token', 'HUGGINGFACE_HUB_TOKEN']}]} 

    # Call loader which should set env var from the mocked Secret Manager
    mod.load_secrets(mod.CONFIG)

    assert os.environ.get('HUGGINGFACE_HUB_TOKEN') == 'hf_token_from_sm'
