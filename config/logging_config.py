import logging.config
import sys
import time

LOGGING_LINE_SIZE = 100

class PipeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Always format as HH:MM:SS,mmm (with milliseconds)
        ct = self.converter(record.created)
        s = time.strftime("%H:%M:%S", ct)
        msecs = int(record.msecs)
        return f"{s},{msecs:03d}"

    def format(self, record):
        asctime = self.formatTime(record, self.datefmt)
        levelname = f"{record.levelname:<8}"
        asctime = f"{asctime:<14}"
        filename = record.filename
        msg = record.getMessage()

        return f"{asctime} | {levelname} | [{filename}] {msg}"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(filename)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        },
        "pipe": {
            '()': PipeFormatter,
            'datefmt': "%H:%M:%S,%03d",
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

# Detect if running in a terminal (not piped/redirected)
if sys.stdout.isatty():
    LOGGING["handlers"]["stdout"]["formatter"] = "pipe"
else:
    LOGGING["handlers"]["stdout"]["formatter"] = "json"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.config.dictConfig(LOGGING)


# Print a banner and skip lines if running in a terminal
if sys.stdout.isatty():
    print("\n" * 10)
    print("=" * LOGGING_LINE_SIZE)
    print("{:^100}".format("LaxAI Starting Application"))
    print("=" * LOGGING_LINE_SIZE)
