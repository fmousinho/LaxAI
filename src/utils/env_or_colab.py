import os
from dotenv import load_dotenv
from IPython import get_ipython

# With TOML configuration, we can directly use relative path from project root
env_path = '.env'


def load_env_or_colab():
    # Try loading .env file (local dev)
    env_loaded = False
    try:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            env_loaded = True
    except Exception:
        pass

    # If not loaded and running in Colab, use userdata API
    if not env_loaded and 'google.colab' in str(get_ipython()):
        try:
            from google.colab import userdata
            os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = userdata.get('GOOGLE_APPLICATION_CREDENTIALS')
        except Exception as e:
            print(f"Colab userdata API not available: {e}")

# Call this at the top of your module
load_env_or_colab()