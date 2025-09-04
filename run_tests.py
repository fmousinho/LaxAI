#!/usr/bin/env python3
"""
Test runner script that ensures tests run in the correct order:
1. Unit tests (fastest)
2. API tests
3. Integration tests
4. Performance tests
5. End-to-end tests (slowest, run last)
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode

def main():
    """Main test runner function."""
    # Get the Python executable
    python_bin = os.environ.get('PYTHON_EXECUTABLE', sys.executable)

    # Phase 1: Unit tests (fastest)
    print("Phase 1: Running unit tests...")
    cmd1 = [python_bin, '-m', 'pytest', 'tests/unit/', '--tb=short', '-v']
    exit_code1 = run_command(cmd1, "Unit Tests")

    if exit_code1 != 0:
        print(f"\n❌ Phase 1 (Unit) failed with exit code {exit_code1}")
        return exit_code1

    print("\n✅ Phase 1 (Unit) completed successfully!")

    # Phase 2: API tests
    print("\nPhase 2: Running API tests...")
    cmd2 = [python_bin, '-m', 'pytest', 'tests/api/', '--tb=short', '-v']
    exit_code2 = run_command(cmd2, "API Tests")

    if exit_code2 != 0:
        print(f"\n❌ Phase 2 (API) failed with exit code {exit_code2}")
        return exit_code2

    print("\n✅ Phase 2 (API) completed successfully!")

    # Phase 3: Integration tests (excluding e2e)
    print("\nPhase 3: Running integration tests...")
    cmd3 = [python_bin, '-m', 'pytest', 'tests/integration/', '-m', 'not e2e', '--tb=short', '-v']
    exit_code3 = run_command(cmd3, "Integration Tests (excluding E2E)")

    if exit_code3 != 0:
        print(f"\n❌ Phase 3 (Integration) failed with exit code {exit_code3}")
        return exit_code3

    print("\n✅ Phase 3 (Integration) completed successfully!")

    # Phase 4: Performance tests
    print("\nPhase 4: Running performance tests...")
    cmd4 = [python_bin, '-m', 'pytest', 'tests/performance/', '--tb=short', '-v']
    exit_code4 = run_command(cmd4, "Performance Tests")

    if exit_code4 != 0:
        print(f"\n❌ Phase 4 (Performance) failed with exit code {exit_code4}")
        return exit_code4

    print("\n✅ Phase 4 (Performance) completed successfully!")

    # Phase 5: End-to-end tests (slowest, run last)
    print("\nPhase 5: Running end-to-end tests...")
    cmd5 = [python_bin, '-m', 'pytest', '-m', 'e2e', '--tb=short', '-v']
    exit_code5 = run_command(cmd5, "End-to-End Tests")

    if exit_code5 != 0:
        print(f"\n❌ Phase 5 (E2E) failed with exit code {exit_code5}")
        return exit_code5

    print("\n✅ All tests completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
