"""
Shared test utilities and fixtures for the LaxAI project.
"""
import os
import sys
from pathlib import Path


def setup_test_paths():
    """
    Set up Python paths for test files that need to import from shared_libs and other modules.

    This function ensures that:
    - The project root is in sys.path
    - The shared_libs directory is in sys.path
    - The services directory is in sys.path

    This is necessary when running tests from VSCode testing browser or other environments
    where the PYTHONPATH may not be properly configured.
    """
    # Find the project root (assuming this file is in tests/conftest.py)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up two levels from tests/

    # Define important paths
    shared_libs_path = project_root / "shared_libs"
    services_path = project_root / "services"

    # Add paths to sys.path if not already present
    paths_to_add = [project_root, shared_libs_path, services_path]

    for path in paths_to_add:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


# Automatically set up paths when this module is imported
setup_test_paths()
