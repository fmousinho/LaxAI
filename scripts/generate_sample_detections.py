#!/usr/bin/env python3
"""
Wrapper script to generate sample detections.

This script calls the actual generation script located in the test data directory.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the detection generation script from the test data directory."""
    project_root = Path(__file__).resolve().parent.parent

    # Path to the actual script
    script_path = project_root / "tests" / "services" / "tracking" / "test_data" / "generate_sample_detections.py"

    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        return 1

    # Run the script from the project root
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, cwd=str(project_root))

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())