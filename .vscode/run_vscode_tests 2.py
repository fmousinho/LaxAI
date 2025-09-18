# .vscode/run_vscode_tests.py
import sys
import os
from pathlib import Path
import pytest

def main():
    """
    This script is a wrapper for running pytest within VS Code.

    It solves environment issues by:
    1.  Constructing ABSOLUTE paths to all source directories. This is critical
        for projects located in paths with spaces (like 'Mobile Documents').
    2.  Programmatically setting the PYTHONPATH environment variable.
    3.  Executing pytest with the arguments passed by the VS Code test extension.
    """
    # Get the project root (the directory containing the .vscode folder).
    project_root = Path(__file__).resolve().parent.parent

    # Define all source directories relative to the project root.
    source_dirs = [
        "shared_libs",
        "services/service_api/src",
        "services/service_cr_train_proxy/src",
        "services/service_tracking/src",
        "services/service_training/src",
    ]

    # Create absolute paths for each source directory.
    absolute_source_paths = [str(project_root / src_dir) for src_dir in source_dirs]

    # Build the PYTHONPATH string, separated by the OS-specific separator.
    python_path = os.pathsep.join(absolute_source_paths)

    # Set the PYTHONPATH environment variable for the pytest subprocess.
    os.environ["PYTHONPATH"] = python_path

    print(f"VS Code Test Runner: Set PYTHONPATH to: {python_path}")
    print(f"VS Code Test Runner: Running pytest with args: {sys.argv[1:]}")

    # Launch pytest with the arguments provided by VS Code.
    # The script's own name (sys.argv[0]) is excluded.
    sys.exit(pytest.main(sys.argv[1:]))

if __name__ == "__main__":
    main()
