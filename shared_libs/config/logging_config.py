import logging.config
import sys
import warnings
import os

LOGGING_LINE_SIZE = 110


def _is_notebook() -> bool:
    """Check if the code is running in a Jupyter-like environment."""
    try:
        # Check for Google Colab
        if "google.colab" in sys.modules:
            return True
        # Check for Jupyter
        try:
            from IPython.core.getipython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell == "ZMQInteractiveShell"
        except ImportError:
            return False
    except ImportError:
        return False


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
        "pipe": {
            "format": "| %(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json",
        }
    },
    "loggers": {"": {"handlers": ["stdout"], "level": "INFO"}},
}

# Detect if running in a terminal (not piped/redirected) or a notebook
if sys.stdout.isatty() or _is_notebook():
    LOGGING["handlers"]["stdout"]["formatter"] = "pipe"
else:
    LOGGING["handlers"]["stdout"]["formatter"] = "json"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.config.dictConfig(LOGGING)

# Setup Google Cloud Logging if in cloud environment
if os.getenv('K_SERVICE') or os.getenv('GAE_ENV') or os.getenv('ENV_TYPE') == 'gcp':
    try:
        from google.cloud import logging as cloud_logging
        client = cloud_logging.Client()
        client.setup_logging(log_level=logging.INFO)
        
        # Ensure framework loggers propagate to root
        framework_loggers = [
            "uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi", "uvicorn.lifespan",
            "gunicorn", "gunicorn.error", "gunicorn.access", "fastapi",
        ]
        for name in framework_loggers:
            lg = logging.getLogger(name)
            lg.handlers = []
            lg.propagate = True
            lg.setLevel(logging.INFO)
                
    except ImportError:
        pass


def print_banner() -> None:
    """Print the application banner."""
    if sys.stdout.isatty():
        print("\n" * 10)
        print("=" * LOGGING_LINE_SIZE)
        print("{:^100}".format("LaxAI Starting Application"))
        print("=" * LOGGING_LINE_SIZE)
