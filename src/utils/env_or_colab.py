import os
import logging
from dotenv import load_dotenv
from IPython import get_ipython

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
        env_path = os.environ.get('CONFIG_TOML_PATH') or os.environ.get('CONFIG_PATH')
        if env_path:
            searched_paths.append(env_path)
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    config = toml.load(f)
                logger.debug(f"✅ Loaded configuration from (env) {env_path}")
                return config

        # 2) Walk up from current working directory
        cwd = os.getcwd()
        cur = cwd
        while True:
            candidate = os.path.join(cur, 'config.toml')
            searched_paths.append(candidate)
            if os.path.exists(candidate):
                with open(candidate, 'r') as f:
                    config = toml.load(f)
                logger.debug(f"✅ Loaded configuration from {candidate}")
                return config
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

        # 3) Package-relative (preserve original behavior)
        package_based = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.toml')
        searched_paths.append(package_based)
        if os.path.exists(package_based):
            with open(package_based, 'r') as f:
                config = toml.load(f)
            logger.debug(f"✅ Loaded configuration from package-relative {package_based}")
            return config

        # 4) Try simpler relatives (one/two levels)
        simpler = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.toml'),
            os.path.join(os.path.dirname(__file__), 'config.toml')
        ]
        for candidate in simpler:
            searched_paths.append(candidate)
            if os.path.exists(candidate):
                with open(candidate, 'r') as f:
                    config = toml.load(f)
                logger.debug(f"✅ Loaded configuration from {candidate}")
                return config

        # If we reach here, nothing was found
        raise FileNotFoundError(f"Configuration file not found. Searched locations:\n" + "\n".join(searched_paths))

    except ImportError:
        raise ImportError("toml package is required. Install with: pip install toml")
    except FileNotFoundError:
        # Re-raise FileNotFoundError so caller can distinguish missing config
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from config.toml: {e}")

# Load config at module level - will raise exception if not available
CONFIG = load_config()

# With TOML configuration, we can directly use relative path from project root
env_path = '.env'


def get_secret_from_manager(secret_name, project_id=None, version="latest"):
    """
    Retrieve a secret from Google Secret Manager.
    
    Args:
        secret_name (str): Name of the secret in Secret Manager
        project_id (str, optional): GCP Project ID. If None, will try to detect automatically
        version (str): Version of the secret to retrieve (default: "latest")
    
    Returns:
        str: The secret value, or None if not found/accessible
    """
    try:
        from google.cloud import secretmanager

        if not project_id:
            # Prefer explicit env var; fall back to config if provided.
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GCLOUD_PROJECT')

        if not project_id:
            logger.debug(f"Cannot retrieve secret '{secret_name}': project_id not provided")
            return None

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        logger.debug(f"✅ Retrieved secret '{secret_name}' from Secret Manager")
        return secret_value

    except ImportError:
        logger.debug("google-cloud-secret-manager not available")
        return None
    except Exception as e:
        logger.debug(f"Failed to retrieve secret '{secret_name}' from Secret Manager: {e}")
        return None


def setup_secret_from_manager(env_var_name, secret_name, project_id=None):
    """
    Set an environment variable from a Secret Manager secret.
    
    Args:
        env_var_name (str): Name of the environment variable to set
        secret_name (str): Name of the secret in Secret Manager
        project_id (str, optional): GCP Project ID
    
    Returns:
        bool: True if secret was successfully retrieved and set, False otherwise
    """
    if os.environ.get(env_var_name):
        logger.debug(f"Environment variable {env_var_name} already set, skipping Secret Manager")
        return True
    
    secret_value = get_secret_from_manager(secret_name, project_id)
    if secret_value:
        os.environ[env_var_name] = secret_value
        logger.info(f"✅ Set {env_var_name} from Secret Manager secret '{secret_name}'")
        return True
    else:
        logger.debug(f"Could not retrieve {env_var_name} from Secret Manager secret '{secret_name}'")
        return False


def is_running_in_colab():
    """Check if code is running in Google Colab."""
    try:
        return 'google.colab' in str(get_ipython())
    except:
        return False


def is_running_in_gcp():
    """
    Check if code is running in Google Cloud Platform.
    This includes Cloud Run, Compute Engine, GKE, Cloud Functions, etc.
    """
    gcp_indicators = [
        'GOOGLE_CLOUD_PROJECT',
        'GCLOUD_PROJECT',
        'K_SERVICE',  # Cloud Run
        'K_REVISION', # Cloud Run
        'FUNCTION_NAME',  # Cloud Functions
        'GAE_SERVICE',  # App Engine
    ]

    for indicator in gcp_indicators:
        if os.environ.get(indicator):
            logger.debug(f"Detected GCP environment via {indicator}")
            return True

    return False


def setup_gcp_credentials():
    """
    Set up credentials for Google Cloud Platform environments.
    In GCP, credentials are typically handled through:
    1. Workload Identity (GKE)
    2. Service Account attached to compute instance
    3. Default Application Credentials
    4. Environment variables already set by the platform
    5. Google Secret Manager for sensitive configuration
    """
    logger.info("Detected GCP environment; relying on Application Default Credentials (ADC)")

    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        logger.info(f"Using explicit service account file: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
    else:
        logger.info("No GOOGLE_APPLICATION_CREDENTIALS set; using platform-provided ADC (Workload Identity / attached SA)")

    # Set GOOGLE_CLOUD_PROJECT from env if available; avoid network metadata calls here.
    gcproj = os.environ.get('GCLOUD_PROJECT')
    if not os.environ.get('GOOGLE_CLOUD_PROJECT') and gcproj:
        os.environ['GOOGLE_CLOUD_PROJECT'] = gcproj
        logger.debug(f"Set GOOGLE_CLOUD_PROJECT from GCLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")

    # Do NOT auto-fetch secrets here. Provide helper functions for explicit use.


def setup_colab_credentials():
    """Set up credentials for Google Colab environment."""
    logger.info("📓 Detected Google Colab environment")
    try:
        import importlib
        colab = importlib.import_module('google.colab')
        userdata = getattr(colab, 'userdata', None)

        # Try to load secrets from Colab userdata first
        secrets_config = CONFIG.get('secrets', {})
        for env_var_name, secret_names in secrets_config.items():
            try:
                if userdata is not None:
                    secret_value = userdata.get(env_var_name)
                else:
                    secret_value = None
                if secret_value:
                    os.environ[env_var_name] = secret_value
                    logger.info(f"✅ {env_var_name} loaded from Colab userdata")
                else:
                    raise RuntimeError("no userdata value")
            except Exception as e:
                logger.warning(f"⚠️  Could not load {env_var_name} from Colab userdata: {e}")
                # Try to fall back to Secret Manager if available
                logger.info(f"🔐 Attempting to retrieve {env_var_name} from Secret Manager...")
                success = False
                for secret_name in secret_names:
                    success = setup_secret_from_manager(env_var_name, secret_name)
                    if success:
                        break

        # Set Google credentials (special case, not in secrets config)
        try:
            if userdata is not None:
                gcp_creds = userdata.get('GOOGLE_APPLICATION_CREDENTIALS')
            else:
                gcp_creds = None
            if gcp_creds:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_creds
                logger.info("✅ Google Application Credentials loaded from Colab userdata")
            else:
                raise RuntimeError("no userdata credentials")
        except Exception as e:
            logger.warning(f"⚠️  Could not load GOOGLE_APPLICATION_CREDENTIALS from Colab userdata: {e}")
            
    except ImportError:
        logger.error("❌ google.colab module not available in Colab environment")
        # Fall back to Secret Manager if colab is not available
        _fallback_to_secret_manager()
    except Exception as e:
        logger.error(f"❌ Failed to load Colab userdata: {e}")
        # Fall back to Secret Manager
        _fallback_to_secret_manager()


def _fallback_to_secret_manager():
    """Helper function to load all configured secrets from Secret Manager."""
    logger.info("🔐 Attempting to retrieve credentials from Secret Manager...")
    secrets_config = CONFIG.get('secrets', {})
    for env_var_name, secret_names in secrets_config.items():
        if not os.environ.get(env_var_name):
            success = False
            for secret_name in secret_names:
                success = setup_secret_from_manager(env_var_name, secret_name)
                if success:
                    break


def setup_local_credentials():
    """Set up credentials for local development environment."""
    logger.info("💻 Detected local development environment")
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"✅ Loaded environment variables from {env_path}")
        
        # Check which configured secrets are missing and try Secret Manager as fallback
        secrets_config = CONFIG.get('secrets', {})
        missing_secrets = []
        
        for env_var_name, secret_names in secrets_config.items():
            if os.environ.get(env_var_name):
                logger.debug(f"✅ {env_var_name} found")
            else:
                logger.warning(f"⚠️  {env_var_name} not found in .env file")
                missing_secrets.append((env_var_name, secret_names))
        
        # Try Secret Manager for missing secrets if we have GCP credentials
        if missing_secrets and os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.info("🔐 Attempting to retrieve missing credentials from Secret Manager...")
            for env_var_name, secret_names in missing_secrets:
                success = False
                for secret_name in secret_names:
                    success = setup_secret_from_manager(env_var_name, secret_name)
                    if success:
                        break
            
        # Check Google credentials
        if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if creds_path and os.path.exists(creds_path):
                logger.debug(f"✅ Google credentials file found: {creds_path}")
            else:
                logger.warning(f"⚠️  Google credentials file not found: {creds_path}")
        else:
            logger.warning("⚠️  GOOGLE_APPLICATION_CREDENTIALS not found in .env file")
    else:
        logger.warning(f"⚠️  Environment file not found: {env_path}")
        # If we have GCP setup but no .env, try Secret Manager
        if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') or os.environ.get('GOOGLE_CLOUD_PROJECT'):
            _fallback_to_secret_manager()


def ensure_credentials():
    """Detect environment and configure credentials accordingly.

    - On GCP: rely on Application Default Credentials (do not set GOOGLE_APPLICATION_CREDENTIALS).
    - On Colab: attempt to load Colab userdata, falling back to Secret Manager.
    - Local: load .env and optionally use Secret Manager for missing values.
    """
    try:
        if is_running_in_gcp():
            setup_gcp_credentials()
        elif is_running_in_colab():
            setup_colab_credentials()
        else:
            setup_local_credentials()
    except Exception as e:
        logger.debug(f"Credential setup encountered an error: {e}")


# Auto-run credential setup by default. Set SKIP_AUTO_CREDENTIAL_SETUP=1 to disable.
if os.environ.get('SKIP_AUTO_CREDENTIAL_SETUP', '').lower() not in ('1', 'true', 'yes'):
    ensure_credentials()


def get_environment_info():
    """
    Get information about the current environment.
    
    Returns:
        dict: Environment information including type, project, and available credentials
    """
    env_info = {
        'type': 'unknown',
        'project_id': os.environ.get('GOOGLE_CLOUD_PROJECT'),
        'has_wandb_key': bool(os.environ.get('WANDB_API_KEY')),
        'has_gcp_credentials': bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')),
        'gcp_metadata_available': False,
        'secret_manager_available': False
    }
    
    # Check if Secret Manager is available
    try:
        from google.cloud import secretmanager
        env_info['secret_manager_available'] = True
    except ImportError:
        pass
    
    if is_running_in_colab():
        env_info['type'] = 'colab'
    elif is_running_in_gcp():
        env_info['type'] = 'gcp'
        # Check metadata availability
        try:
            import requests
            response = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/',
                headers={'Metadata-Flavor': 'Google'},
                timeout=2
            )
            env_info['gcp_metadata_available'] = response.status_code == 200
        except ImportError:
            logger.debug("requests module not available for metadata check")
        except Exception:
            pass
    else:
        env_info['type'] = 'local'
    
    return env_info


def load_env_or_colab():
    """
    Load environment variables and credentials based on the detected environment.
    Supports:
    1. Local development (with .env files and optional Secret Manager fallback)
    2. Google Colab (with userdata API and Secret Manager fallback)  
    3. Google Cloud Platform (with metadata service, default credentials, and Secret Manager)
    """
    try:
        if is_running_in_colab():
            setup_colab_credentials()
        elif is_running_in_gcp():
            setup_gcp_credentials()
        else:
            setup_local_credentials()
            
        # Log environment summary
        env_info = get_environment_info()
        logger.info(f"🚀 Environment setup complete:")
        logger.info(f"   Type: {env_info['type']}")
        logger.info(f"   Project: {env_info['project_id'] or 'Not detected'}")
        logger.info(f"   WandB: {'✅' if env_info['has_wandb_key'] else '❌'}")
        logger.info(f"   GCP Auth: {'✅' if env_info['has_gcp_credentials'] or env_info['type'] == 'gcp' else '❌'}")
        logger.info(f"   Secret Manager: {'✅' if env_info['secret_manager_available'] else '❌'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load environment: {e}")
        # Continue anyway - some functionality might still work

# Call this at the top of your module
load_env_or_colab()