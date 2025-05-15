
import os
import sys 
from . import constants as const
import logging
from .store_driver import Store
import atexit
from . import application as app
import torch 
from torch.utils.tensorboard import SummaryWriter
import importlib.metadata # For check_requirements
from packaging.requirements import Requirement, InvalidRequirement # For check_requirements
from packaging.version import Version # For check_requirements
import datetime

# Determine the directory of the main.py script first
# This is needed for resolving paths for .env and log_dir
_MAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DOTENV_PATH = os.path.join(_MAIN_SCRIPT_DIR, ".env")

def check_for_gpu() -> torch.device:
    """Checks if a GPU is available and returns the appropriate torch device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the first available GPU
        logger.info("GPU is available")
    else:
        device = torch.device("cpu")  # Use the CPU if no GPU is available
        logger.info("GPU is not available, using CPU instead.")
    return device

#--- Initialize logging
log_dir = os.path.join(_MAIN_SCRIPT_DIR, "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s (%(name)s)')
logger = logging.getLogger(__name__) # Get logger for main.py
logger.info(f"TensorBoard logging to: {log_dir}")
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
    if writer:
        writer.close()

atexit.register(cleanup_temp_files)

# --- Requirements Check Function (Moved from initialize.py) ---

def check_requirements(requirements_filename="requirements.txt"):
    """
    Checks if packages listed in the requirements file are installed.

    Args:
        requirements_filename: Name of the requirements file (e.g., "requirements.txt").
                               This file is expected to be in the same directory as this script.

    Returns:
        True if all requirements are met, False otherwise.
    """
    # Construct the absolute path to the requirements file
    # __file__ is the path to the current script (main.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, requirements_filename)
    try:
        missing_packages = []
        version_mismatches = []
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines and comments
                    continue
                try:
                    req = Requirement(line)
                    dist = importlib.metadata.distribution(req.name)
                    installed_version = Version(dist.version)
                    if not req.specifier.contains(installed_version, prereleases=True):
                        version_mismatches.append(f"  - {req.name}: Installed={installed_version}, Required={req.specifier}")
                except InvalidRequirement:
                    logger.error(f"Warning: Skipping invalid requirement line: {line}")
                except importlib.metadata.PackageNotFoundError:
                    missing_packages.append(f"  - {req.name} ({req.specifier or 'any version'})")

        if not missing_packages and not version_mismatches: # All good
            logger.info("All requirements are installed and versions match.")
            return True
    except FileNotFoundError:
        logger.error(f"{requirements_path} not found.")
        return False

    # Report errors if any were found
    if missing_packages:
        logger.error("Error: The following required packages are missing:")
        logger.error("\n".join(missing_packages))
    if version_mismatches:
        logger.error("Error: The following installed packages have version mismatches:")
        logger.error("\n".join(version_mismatches))

    if missing_packages or version_mismatches:
        logger.error(f"\nPlease install or update the required packages. If running from the project root, you might use: pip install -r {requirements_filename}")
        return False

    return True # Should have returned True if no errors and no FileNotFoundError

# --- Perform Requirements Check ---
if not check_requirements(): # Call the local check_requirements function
    sys.exit(1)

# --- Environment Variables Setup ---
logger.info("Loading environment variables.")
try:
    from dotenv import load_dotenv
    if load_dotenv(dotenv_path=_DOTENV_PATH): # Use the explicit path
        logger.info(f".env file loaded successfully from {_DOTENV_PATH}.")
    else:
        logger.warning(f".env file found at {_DOTENV_PATH} but could not be loaded (e.g., empty or parsing error).")
except ImportError:
    logger.warning("python-dotenv not installed. Cannot load .env file.")

# --- Check for GPU availability ---
device = check_for_gpu()

# --- Google Storage  ---
store = Store(
    temp_file_registrar=register_temp_file,
    cache_dir=const.CACHE_DIR,
    cache_duration=const.CACHE_DURATION_SECONDS
)
if store.service:
    app.run_application(store=store, writer=writer, device=device) # Pass device
else:
    logger.critical("Failed to initialize storage service (authentication likely failed). Cannot start application.")
    sys.exit(1) 
