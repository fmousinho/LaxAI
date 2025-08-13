import logging.config
import sys
import time


LOGGING_LINE_SIZE = 110

def _is_notebook() -> bool:
    """Check if the code is running in a Jupyter-like environment."""
    try:
        # Check for Google Colab
        if 'google.colab' in sys.modules:
            return True
        # Check for Jupyter, an 'ipython' console does not count.
        # get_ipython is a builtin in IPython environments.
        shell = get_ipython().__class__.__name__  
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook, JupyterLab, qtconsole
        return False
    except NameError:
        return False      # Not in an IPython-like environment

class PipeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Always format as HH:MM:SS,mmm (with milliseconds)
        ct = self.converter(record.created)
        s = time.strftime("%H:%M:%S", ct)
        msecs = int(record.msecs)
        return f"{s},{msecs:03d}"

    def format(self, record):
        asctime = self.formatTime(record, self.datefmt)
        levelname = f"{record.levelname:<7}"
        asctime = f"{asctime:<12}"
        # Remove extension from filename
        filename = record.filename.rsplit('.', 1)[0] if '.' in record.filename else record.filename
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

# Detect if running in a terminal (not piped/redirected) or a notebook
if sys.stdout.isatty() or _is_notebook():
    LOGGING["handlers"]["stdout"]["formatter"] = "pipe"
else:
    LOGGING["handlers"]["stdout"]["formatter"] = "json"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.config.dictConfig(LOGGING)


def _check_if_main_and_show_banner():
    """
    Automatically detect if we're being imported by a main module and show banner.
    This runs automatically when logging_config is imported.
    """
    import inspect
    
    # Get the current frame
    frame = inspect.currentframe()
    if frame is None:
        return
        
    try:
        # Look up the call stack to find if any caller is __main__
        caller_frame = frame.f_back
        while caller_frame:
            caller_name = caller_frame.f_globals.get('__name__', '')
            if caller_name == '__main__':
                # Only show banner if running in a terminal and called from main
                if sys.stdout.isatty():
                    print("\n" * 10)
                    print("=" * LOGGING_LINE_SIZE)
                    print("{:^100}".format("LaxAI Starting Application"))
                    print("=" * LOGGING_LINE_SIZE)
                return
            caller_frame = caller_frame.f_back
    finally:
        del frame



# Automatically check and show banner when this module is imported
_check_if_main_and_show_banner()
