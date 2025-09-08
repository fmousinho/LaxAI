#!/usr/bin/env python3
"""
Multi-Service Test Runner for LaxAI
Runs tests across all services in the correct order:
1. Service unit tests (fastest)
2. Service integration tests
3. Cross-service integration tests
4. Performance tests
5. End-to-end tests (slowest, run last)
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    if cwd:
        print(f"Directory: {cwd}")
    print('='*60)

    result = subprocess.run(cmd, cwd=cwd or os.getcwd())
    return result.returncode

def run_service_tests(service_name, python_bin):
    """Run tests for a specific service."""
    service_path = Path(f"services/{service_name}")
    if not service_path.exists():
        print(f"⚠️  Service {service_name} not found, skipping...")
        return 0

    venv_python = service_path / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        print(f"⚠️  Virtual environment not found for {service_name}, using system Python")
        python_cmd = python_bin

    # Check if tests directory exists
    tests_path = service_path / "tests"
    if not tests_path.exists():
        print(f"ℹ️  No tests directory found for {service_name}, skipping...")
        return 0

    # Run unit tests for the service
    exit_code = run_command(
        [python_cmd, "-m", "pytest", "tests/", "-v", "-m", "not integration"],
        f"Unit tests for {service_name}",
        cwd=service_path
    )
    
    return exit_code

def main():
    """Main test runner function."""
    # Get the Python executable
    python_bin = os.environ.get('PYTHON_EXECUTABLE', sys.executable)
    
    services = ['service-tracking', 'service-training', 'service-api', 'service-cloud']
    failed_tests = []

    print("🧪 Starting Multi-Service Test Suite")
    print("=" * 60)

    # Phase 1: Service unit tests (fastest)
    print(f"\n🏃 Phase 1: Service Unit Tests")
    for service in services:
        print(f"\n📦 Testing {service}")
        exit_code = run_service_tests(service, python_bin)
        if exit_code != 0:
            failed_tests.append(f"{service} unit tests")

    # Phase 2: Cross-service integration tests
    print(f"\n🔗 Phase 2: Integration Tests")
    if Path("tests").exists():
        exit_code = run_command(
            [python_bin, "-m", "pytest", "tests/integration/", "-v"],
            "Cross-service integration tests",
            cwd="."
        )
        if exit_code != 0:
            failed_tests.append("Integration tests")

    # Phase 3: Performance tests
    print(f"\n⚡ Phase 3: Performance Tests")
    if Path("tests/performance").exists():
        exit_code = run_command(
            [python_bin, "-m", "pytest", "tests/performance/", "-v", "-m", "performance"],
            "Performance tests",
            cwd="."
        )
        if exit_code != 0:
            failed_tests.append("Performance tests")

    # Phase 4: End-to-end tests (slowest)
    print(f"\n🎯 Phase 4: End-to-End Tests")
    if Path("tests").exists():
        exit_code = run_command(
            [python_bin, "-m", "pytest", "tests/", "-v", "-m", "e2e"],
            "End-to-end tests",
            cwd="."
        )
        if exit_code != 0:
            failed_tests.append("End-to-end tests")

    # Summary
    print(f"\n{'='*60}")
    print("🏁 Test Summary")
    print('='*60)
    
    if failed_tests:
        print("❌ Some tests failed:")
        for test in failed_tests:
            print(f"   - {test}")
        return 1
    else:
        print("✅ All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
