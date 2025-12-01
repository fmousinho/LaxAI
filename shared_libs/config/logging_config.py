import logging.config
import sys
import warnings
import os
import json

LOGGING_LINE_SIZE = 110


class GCPJSONFormatter(logging.Formatter):
    """
    JSON formatter for Google Cloud Logging with structured fields.
    
    Outputs logs as JSON with 'severity' field (GCP standard) and makes
    all fields searchable in Cloud Logging console.
    """
    
    # Standard logging record attributes to exclude from extra fields
    RESERVED_ATTRS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName', 
        'levelname', 'levelno', 'lineno', 'module', 'msecs', 
        'message', 'pathname', 'process', 'processName', 'relativeCreated',
        'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
        'taskName', 'asctime'
    }
    
    def format(self, record):
        """Format log record as JSON for GCP."""
        log_obj = {
            "severity": record.levelname,  # GCP uses "severity" for log level
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields from extra={} parameter
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS:
                # Ensure value is JSON serializable
                try:
                    json.dumps(value)
                    log_obj[key] = value
                except (TypeError, ValueError):
                    log_obj[key] = str(value)
        
        return json.dumps(log_obj)


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


def _is_gcp_environment() -> bool:
    """Check if running in Google Cloud Platform environment."""
    return (
        os.getenv('K_SERVICE') is not None or  # Cloud Run
        os.getenv('GAE_ENV') is not None or     # App Engine
        os.getenv('ENV_TYPE') == 'gcp' or       # Custom GCP indicator
        os.getenv('GOOGLE_CLOUD_PROJECT') is not None  # Any GCP service
    )


# Get log level from environment, default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
        "gcp_json": {
            "()": GCPJSONFormatter,  # Use custom formatter class
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
    "loggers": {"": {"handlers": ["stdout"], "level": LOG_LEVEL}},
}

# Detect environment and set appropriate formatter
if sys.stdout.isatty() or _is_notebook():
    LOGGING["handlers"]["stdout"]["formatter"] = "pipe"
elif _is_gcp_environment():
    LOGGING["handlers"]["stdout"]["formatter"] = "gcp_json"  # Use structured JSON for GCP
else:
    LOGGING["handlers"]["stdout"]["formatter"] = "json"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


"""Logging configuration

In Cloud Run/GCP, prefer stdout/stderr for log collection. Attempt to set up
google-cloud-logging as a secondary path, but never remove stdout handlers to
avoid dropping logs if the API client fails or lacks permissions.
"""

is_cloud_env = os.getenv('K_SERVICE') or os.getenv('GAE_ENV') or os.getenv('ENV_TYPE') == 'gcp'

if is_cloud_env:
    try:
        from google.cloud import logging as cloud_logging  # type: ignore
        client = cloud_logging.Client()
        # Map string level to logging constant
        log_level_const = getattr(logging, LOG_LEVEL, logging.INFO)
        client.setup_logging(log_level=log_level_const)
    except Exception:
        # If Cloud Logging client isn't available or fails, silently fall back to stdout
        # We still configure stdout below.
        pass

# Always configure logging with our dictConfig (keep stdout handler intact)
logging.config.dictConfig(LOGGING)

# Ensure framework loggers propagate to root and don't carry duplicate handlers
framework_loggers = [
    "uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi", "uvicorn.lifespan",
    "gunicorn", "gunicorn.error", "gunicorn.access", "fastapi",
]
for name in framework_loggers:
    lg = logging.getLogger(name)
    lg.propagate = True
    # Don't forcibly clear handlers here; let dictConfig manage root handlers
    # and avoid stripping handlers that frameworks may rely on.
    if is_cloud_env:
        lg.setLevel(LOG_LEVEL)


def print_banner() -> None:
    """Print the application banner."""
    if sys.stdout.isatty():
        print("\n" * 10)
        print("=" * LOGGING_LINE_SIZE)
        print("{:^100}".format("LaxAI Starting Application"))
        print("=" * LOGGING_LINE_SIZE)
