import sys
from pathlib import Path
import os

# This file is crucial for ensuring tests run correctly in VS Code and from the CLI.
# It programmatically modifies the Python path to include all necessary source directories.

# Get the absolute path of the project root directory.
# Path(__file__).resolve() gets the full path to this conftest.py file.
# .parent gives the directory containing it, which is the project root.
project_root = Path(__file__).resolve().parent

# Define a list of all directories that contain Python source code.
# These paths are relative to the project root.
source_dirs = [
    "shared_libs",
    "services/service_api/src",
    "services/service_cr_train_proxy/src",
    "services/service_dataprep",
    "services/service_dataprep/src",
    "services/service_tracking/src",
    "services/service_training/src",
]

# Convert the relative paths to absolute paths.
# This is critical for ensuring they are resolved correctly, especially in environments
# where the current working directory might be inconsistent (like VS Code test discovery).
absolute_source_paths = [str(project_root / src_dir) for src_dir in source_dirs]

# Get the existing PYTHONPATH, if it's already set.
existing_python_path = os.environ.get("PYTHONPATH", "")

# Split the existing PYTHONPATH by the OS-specific separator (':' on Unix, ';' on Windows).
# Filter out any empty strings that might result from splitting.
existing_paths = [p for p in existing_python_path.split(os.pathsep) if p]

# For absolute imports from the workspace root, we need the project root in PYTHONPATH
# Combine the project root with the existing paths.
combined_paths = set([str(project_root)] + existing_paths)

# Join the unique paths back into a single string using the OS separator.
new_python_path = os.pathsep.join(sorted(list(combined_paths)))

# Set the updated PYTHONPATH environment variable.
# This will be used by Python to find modules during import.
os.environ["PYTHONPATH"] = new_python_path
# Also update sys.path directly for the current process.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set environment variables for tests
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "laxai-466119")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

print(f"Updated PYTHONPATH: {new_python_path}")
