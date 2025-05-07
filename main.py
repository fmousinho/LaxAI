
import os
import sys
import initialize
import constants as const
import logging
from store_driver import Store
import atexit
import os
import application as app

#--- Initialize logging

logging.basicConfig(level=logging.INFO, format=': %(asctime)s %(levelname)s %(message)s (%(name)s)')
logger = logging.getLogger(__name__) # Get logger for main.py
logger.info("--- Logger initialized in main.py ---")

# --- Temporary File Management ---
_temp_files_registry = set()

def register_temp_file(path: str):
    """Adds a temporary file path to the registry for cleanup."""
    if path:
        logger.debug(f"Registering temporary file for cleanup: {path}")
        _temp_files_registry.add(path)

def cleanup_temp_files():
    """Deletes all registered temporary files."""
    logger.info("Performing application cleanup...")
    for path in list(_temp_files_registry):
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Successfully deleted temporary file: {path}")
                _temp_files_registry.remove(path)
        except OSError as e:
            logger.error(f"Error deleting temporary file {path}: {e}", exc_info=True)

atexit.register(cleanup_temp_files)

if not initialize.check_requirements():
    sys.exit(1) 

# --- Environment Variables Setup ---
logger.info("Loading environment variables.")
try:
    from dotenv import load_dotenv
    if load_dotenv():
        logger.info(".env file loaded successfully.")
    else:
        logger.warning(".env file not found or could not be loaded.")
except ImportError:
    logger.warning("python-dotenv not installed. Cannot load .env file.")

# --- Google Storage  ---
store = Store(
    temp_file_registrar=register_temp_file,
    cache_dir=const.CACHE_DIR,
    cache_duration=const.CACHE_DURATION_SECONDS
)
if store.service:
    app.run_application(store)
else:
    logger.critical("Failed to initialize storage service (authentication likely failed). Cannot start application.")
    sys.exit(1) 
