import logging.config
import sys
import time
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
            if shell == "ZMQInteractiveShell":
                return True
            return False
        except ImportError:
            return False
    except NameError:
        return False


class PipeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Always format as HH:MM:SS,mmm
        ct = self.converter(record.created)
        s = time.strftime("%H:%M:%S", ct)
        msecs = int(record.msecs)
        return f"{s},{msecs:03d}"

    def format(self, record):
        asctime = self.formatTime(record, self.datefmt)
        levelname = f"{record.levelname:<7}"
        asctime = f"{asctime:<12}"
        # Remove extension from filename
        filename = record.filename.rsplit(".", 1)[0] if "." in record.filename else record.filename
        msg = record.getMessage()

        return f"{asctime} | {levelname} | [{filename}] {msg}"


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(filename)s %(message)s",
            "class": "pythonjsonlogger.json.JsonFormatter",
        },
        "pipe": {
            "()": PipeFormatter,
            "datefmt": "%H:%M:%S,%03d",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            # Formatter will be set dynamically below
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
        # Setup Google Cloud Logging
        client = cloud_logging.Client()
        handler = client.get_default_handler()
        
        # In GCP, use only Google Cloud Logging to prevent duplication
        LOGGING["loggers"][""] = {"handlers": [], "level": "INFO"}
        logging.config.dictConfig(LOGGING)
        
        # Add the Google Cloud handler to the root logger
        root_logger = logging.getLogger()
        # Avoid duplicate handlers on reloads
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        # Ensure framework loggers (uvicorn/gunicorn/fastapi) propagate to root
        # so their messages are captured by Cloud Logging handler.
        framework_loggers = [
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
            "uvicorn.asgi",
            "uvicorn.lifespan",
            "gunicorn",
            "gunicorn.error",
            "gunicorn.access",
            "fastapi",
        ]
        for name in framework_loggers:
            try:
                lg = logging.getLogger(name)
                # Remove existing stream handlers to avoid double stdout logging
                lg.handlers = []
                lg.propagate = True
                lg.setLevel(logging.INFO)
            except Exception:
                # Be defensive—if a logger isn't present yet, skip
                pass
        
    except ImportError:
        # google-cloud-logging not available, continue with stdout logging
        pass
else:
    # For local development, use stdout logging
    logging.config.dictConfig(LOGGING)


def print_banner() -> None:
    """Print the application banner."""
    if sys.stdout.isatty():
        print("\n" * 10)
        print("=" * LOGGING_LINE_SIZE)
        print("{:^100}".format("LaxAI Starting Application"))
        print("=" * LOGGING_LINE_SIZE)
