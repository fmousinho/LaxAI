import logging
import os
from typing import Any, Dict

import requests

try:
    from IPython.core.getipython import get_ipython
except ImportError:
    # IPython not available (e.g., in cloud environments)
    get_ipython = None

logger = logging.getLogger(__name__)

# Load configuration from config.toml - REQUIRED


def load_config():
    """Load configuration from config.toml file.

    The function will attempt to find `config.toml` in several locations in this order:
      1. Path specified via environment variables `CONFIG_TOML_PATH` or `CONFIG_PATH`.
      2. Current working directory and its parent directories (walking up to root).
      3. Project/package relative path (three levels up from this file) - original behavior.
      4. The directory containing this file and its parent.

    If the file cannot be found, a FileNotFoundError is raised listing all attempted locations.
    """
    try:
        import toml

        searched_paths = []

        # 1) Environment override
        env_path = os.environ.get("CONFIG_TOML_PATH")
        if env_path:
            searched_paths.append(env_path)
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    config = toml.load(f)
                logger.debug(f"✅ Loaded configuration from (env) {env_path}")
                return config

        # 2) Walk up from current working directory
        cwd = os.getcwd()
        cur = cwd
        while True:
            candidate = os.path.join(cur, "config.toml")
            searched_paths.append(candidate)
            if os.path.exists(candidate):
                with open(candidate, "r") as f:
                    config = toml.load(f)
                logger.debug(f"✅ Loaded configuration from {candidate}")
                return config
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

        # 3) Package-relative (preserve original behavior)
        package_based = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.toml"
        )
        searched_paths.append(package_based)
        if os.path.exists(package_based):
            with open(package_based, "r") as f:
                config = toml.load(f)
            logger.debug(f"✅ Loaded configuration from package-relative {package_based}")
            return config

        # 4) Try simpler relatives (one/two levels)
        simpler = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml"),
            os.path.join(os.path.dirname(__file__), "config.toml"),
        ]
        for candidate in simpler:
            searched_paths.append(candidate)
            if os.path.exists(candidate):
                with open(candidate, "r") as f:
                    config = toml.load(f)
                logger.debug(f"✅ Loaded configuration from {candidate}")
                return config

        # If we reach here, nothing was found
        raise FileNotFoundError(
            "Configuration file not found. Searched locations:\n" + "\n".join(searched_paths)
        )

    except ImportError:
        raise ImportError("toml package is required. Install with: pip install toml")
    except FileNotFoundError:
        # Re-raise FileNotFoundError so caller can distinguish missing config
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from config.toml: {e}")


# Load config at module level - will raise exception if not available
CONFIG = load_config()

try:
    from dotenv import load_dotenv

    _DOTENV_INSTALLED = True
except ImportError:
    _DOTENV_INSTALLED = False
    logging.warning("python-dotenv not installed. Skipping .env file checks.")


try:
    from google.cloud import secretmanager

    _SECRETS_MANAGER_INSTALLED = True
except ImportError:
    _SECRETS_MANAGER_INSTALLED = False
    logging.warning("google-cloud-secret-manager not installed. Skipping Secret Manager checks.")


def _get_from_env(secret_name: str) -> str | None:
    """Helper to get a secret from environment variables."""
    return os.getenv(secret_name)


def _get_from_dotenv(secret_name: str) -> str | None:
    """Helper to get a secret from a .env file."""
    if not _DOTENV_INSTALLED:
        return None

    load_dotenv()
    return os.getenv(secret_name)


def _get_from_secret_manager(secret_name: str, project_id: str) -> str | None:
    """Helper to get a secret from Google Secret Manager."""
    if not _SECRETS_MANAGER_INSTALLED:
        return None

    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.warning(f"Failed to fetch secret '{secret_name}' from Secret Manager: {e}")
        return None


def load_secrets(config: Dict[str, Any]):
    """
    Loads secrets and sets them as environment variables based on a
    predefined order of precedence.

    1. Existing environment variables
    2. .env file
    3. Google Secret Manager

    Args:
        config: A dictionary with a "secrets" key listing the names of secrets to load.

    Raises:
        ValueError: If a required secret cannot be found.
    """
    secrets_to_load = config.get("secrets", [])
    # If config.toml used a [secrets] table with a `secrets` key, unwrap it.
    if isinstance(secrets_to_load, dict) and "secrets" in secrets_to_load:
        secrets_to_load = secrets_to_load["secrets"]
    # If it's a single string, convert to list
    if isinstance(secrets_to_load, str):
        secrets_to_load = [secrets_to_load]

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logging.warning("GOOGLE_CLOUD_PROJECT is not set. Secret Manager will not work.")

    for secret_name in secrets_to_load:
        # 1) Check existing environment variable
        secret_value = _get_from_env(secret_name)
        if secret_value:
            logging.info(f" ✅ Secret '{secret_name}' found in environment variables.")
            continue

        # 2) Check .env file
        secret_value = _get_from_dotenv(secret_name)
        if secret_value:
            os.environ[secret_name] = secret_value
            logging.info(f" ✅ Secret '{secret_name}' loaded from .env file.")
            continue

        # 3) Check Secret Manager
        if project_id:
            secret_value = _get_from_secret_manager(secret_name, project_id)
            if secret_value:
                os.environ[secret_name] = secret_value
                logging.info(f" ✅ Secret '{secret_name}' loaded from Secret Manager.")
                continue

        # Raise an error if the secret is not found
        raise ValueError(f"Secret '{secret_name}' not found anywhere.")


def is_running_in_colab():
    """Check if code is running in Google Colab."""
    try:
        if get_ipython is None:
            return False
        in_colab = "google.colab" in str(get_ipython())
        logger.info(f"Running in Google Colab: {in_colab}")
        return in_colab
    except Exception:
        return False


def is_running_in_gcp():
    """
    Check if code is running in Google Cloud Platform.
    This includes Cloud Run, Compute Engine, GKE, Cloud Functions, etc.
    """
    metadata_server_url = CONFIG.get(
        "metadata_base_url", "http://169.254.169.254/computeMetadata/v1/"
    )
    headers = {"Metadata-Flavor": "Google"}

    try:
        # A simple request to a known endpoint
        response = requests.get(metadata_server_url, headers=headers, timeout=1)

        # Check if the response is successful
        if response.status_code == 200:
            logger.info("Metadata server check passed: Running in GCP.")
            return True
        else:
            logger.info(f"Metadata server returned status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logger.info(f"Metadata server check failed: {e}")
        return False


def verify_gcp_credentials():
    """
    Validates the presence of key Google Cloud environment variables.

    This function checks for GOOGLE_APPLICATION_CREDENTIALS for explicit
    credential files and GOOGLE_CLOUD_PROJECT for the project ID.
    If a variable is missing, it raises a ValueError.
    """

    # Check for GOOGLE_APPLICATION_CREDENTIALS for explicit credential files
    # This is not strictly required for ADC, but a good practice to validate
    # if you expect a service account key to be present.
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        if not os.path.exists(credentials_path):
            raise ValueError(
                f"GOOGLE_APPLICATION_CREDENTIALS path '{credentials_path}' does not exist."
            )
        logger.info(f"Using explicit service account file: {credentials_path}")
    else:
        logger.info("No GOOGLE_APPLICATION_CREDENTIALS set; relying on default authentication.")

    # Check for GOOGLE_CLOUD_PROJECT, which is a required configuration for most apps.
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        # Fallback check for GCLOUD_PROJECT, a common older variable name
        project_id = os.environ.get("GCLOUD_PROJECT")
        if project_id:
            logger.info("Using GCLOUD_PROJECT as fallback for project ID.")
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id  # set for consistency
        else:
            # Don't raise during import/runtime init. Failures to use GCP APIs will
            # surface later when those APIs are invoked. Log a warning instead so
            # containers won't crash on startup if the env var isn't provided.
            logger.warning(
                "Required environment variable 'GOOGLE_CLOUD_PROJECT' not found."
                " Some GCP features (Secret Manager, Firestore) may not work without it."
            )

    logger.info(f"Google Cloud project ID: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")

    # Do NOT auto-fetch secrets here. Provide helper functions for explicit use.


def set_google_application_credentials():
    """
    Attempts to set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
    Checks if it's already set and points to a valid file.
    If not set, attempts to fall back to default credentials.
    """

    # 1) If already set and file exists, keep it
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        if os.path.exists(cred_path):
            logger.info(f"Using explicit service account file from env: {cred_path}")
            return
        else:
            logger.warning(
                f"GOOGLE_APPLICATION_CREDENTIALS is set but path does not exist: {cred_path}"
            )

    # 2) Try loading from .env (if present). This supports the common workflow
    # where sensitive paths are stored in a local .env file for development.
    try:
        if _DOTENV_INSTALLED:
            load_dotenv()
            env_val = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if env_val:
                if os.path.exists(env_val):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = env_val
                    logger.info(f"Loaded GOOGLE_APPLICATION_CREDENTIALS from .env: {env_val}")
                    return
                else:
                    logger.warning(
                        f"GOOGLE_APPLICATION_CREDENTIALS from .env does not point to a file: {env_val}"
                    )
    except Exception as e:
        logger.debug(f"Error loading .env for GOOGLE_APPLICATION_CREDENTIALS: {e}")

    # 3) As a last-resort, attempt to load from Secret Manager if configured.
    # load_secrets expects a mapping with a 'secrets' key or a list/string; use the canonical form.
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if project_id and _SECRETS_MANAGER_INSTALLED:
        try:
            load_secrets({"secrets": ["GOOGLE_APPLICATION_CREDENTIALS"]})
            # If set via secret manager, verify path exists
            cred_path2 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if cred_path2 and os.path.exists(cred_path2):
                logger.info(
                    f"Loaded GOOGLE_APPLICATION_CREDENTIALS from Secret Manager: {cred_path2}"
                )
                return
            else:
                logger.warning(
                    "GOOGLE_APPLICATION_CREDENTIALS loaded from Secret Manager but file not found or not a path."
                )
        except Exception as e:
            logger.warning(
                f"Could not load GOOGLE_APPLICATION_CREDENTIALS from Secret Manager: {e}"
            )

    # If we reach here, we did not set credentials. Log and continue.
    logger.info(
        "No GOOGLE_APPLICATION_CREDENTIALS found via env, .env, or Secret Manager. Relying on ADC if available."
    )


def setup_environment_secrets():
    """
    Load environment variables and credentials based on the detected environment.
    Supports:
    1. Local development (with .env files and optional Secret Manager fallback)
    2. Google Colab (with userdata API and Secret Manager fallback)
    3. Google Cloud Platform (with metadata service, default credentials, and Secret Manager)
    """
    env_info = dict()
    try:
        if is_running_in_colab():
            set_google_application_credentials()
            env_info["type"] = "colab"
        elif is_running_in_gcp():
            verify_gcp_credentials()
            env_info["type"] = "gcp"
        else:
            set_google_application_credentials()
            env_info["type"] = "local"

        load_secrets(CONFIG.get("secrets", {}))
    except Exception as e:
        logger.error(f"❌ Failed to load environment: {e}")
        raise (e)
        # Continue anyway - some functionality might still work
