# Shared Libraries - Common utilities and base classes
import os
import sys
from pathlib import Path


def setup_shared_libs_paths():
    """
    Set up Python paths to ensure shared_libs and related modules can be imported.

    This function should be called at the beginning of any module that imports
    from shared_libs, especially when running in environments like VSCode testing
    browser where PYTHONPATH may not be properly configured.

    This ensures that:
    - The project root is in sys.path
    - The shared_libs directory is in sys.path
    - The services directory is in sys.path
    """
    # Find the project root (assuming shared_libs is at project root level)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up two levels from shared_libs/

    # Define important paths
    shared_libs_path = current_file.parent  # The shared_libs directory itself
    services_path = project_root / "services"

    # Add paths to sys.path if not already present
    paths_to_add = [project_root, shared_libs_path, services_path]

    for path in paths_to_add:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


# Automatically set up paths when this module is imported
# This ensures that any module importing from shared_libs will have the correct paths
setup_shared_libs_paths()
