import sys
import os
import logging
from . import constants as const
import importlib.metadata
from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version


# --- Basic Logging Setup ---
logger = logging.getLogger(__name__)


# --- Requirements Check Function ---

def check_requirements(requirements_path="requirements.txt"):
    """
    Checks if packages listed in the requirements file are installed.

    Args:
        requirements_path: Path to the requirements file.

    Returns:
        True if all requirements are met, False otherwise.
    """
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

        if not missing_packages and not version_mismatches:
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
        logger.error("\nPlease install or update the required packages using: pip install -r requirements.txt")
        return False

    return False

