#!/usr/bin/env python3
"""
Multi-Service Test Runner for LaxAI
Runs tests across all services in the correct order:
1.     #    # For cross-service imports, include all service src directories
    all_service_src_paths = [str(Path(f"services/{s}/src").resolve()) for s in ['service_tracking', 'service_training', 'service_cloud'] if Path(f"services/{s}/src").exists()]
    
    # Construct PYTHONPATH: shared_libs first, then root directory, then all service src directories
    python_path_parts = [shared_libs_path, root_path] + all_service_src_paths
    service_env["PYTHONPATH"] = ":".join(python_path_parts)ross-service imports, include all service src directories
    all_service_src_paths = [str(Path(f"services/{s}/src").resolve()) for s in ['service_tracking', 'service_training', 'service_cloud'] if Path(f"services/{s}/src").exists()]
    
    # Construct PYTHONPATH: shared_libs first, then root directory, then all service src directories
    python_path_parts = [shared_libs_path, root_path] + all_service_src_paths
    service_env["PYTHONPATH"] = ":".join(python_path_parts)e unit tests
2. Service integration tests
3. Cross-service integration tests
4. Performance tests
5. End-to-end tests (slowest, run last)

Usage:
    python run_tests.py                           # Run all tests
    python run_tests.py --service service_training # Run tests for specific service
    python run_tests.py --category unit           # Run only unit tests
    python run_tests.py --service service_training --category integration  # Run integration tests for specific service
    python run_tests.py --category performance    # Run only performance tests
    python run_tests.py --sv-only                 # Run only System Verification tests
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Service Test Runner for LaxAI")

    parser.add_argument(
        "--service",
        choices=["service_tracking", "service_training", "service_cloud"],
        help="Run tests for a specific service only"
    )

    parser.add_argument(
        "--category",
        choices=["unit", "integration", "performance", "e2e", "all"],
        default="all",
        help="Run tests of a specific category (default: all)"
    )

    parser.add_argument(
        "--sv-only",
        action="store_true",
        help="Run only System Verification (SV) tests (multi-service integration and performance)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def run_command(cmd, description, cwd=".", env=None, verbose=False):
    """Run a command with proper error handling."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Directory: {cwd}")
        print('='*60)
    
    try:
        # Use the provided env or copy the current environment
        command_env = env if env is not None else os.environ.copy()
        
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=False, # Do not raise exception on non-zero exit code
            env=command_env
        )
        return result.returncode
    except FileNotFoundError as e:
        print(f"❌ FileNotFoundError: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

def run_service_tests(service_name, python_bin, category="all", verbose=False):
    """Run tests for a specific service with optional category filtering."""
    service_path = Path(f"services/{service_name}").resolve()
    if not service_path.exists():
        print(f"⚠️  Service {service_name} not found, skipping...")
        return 0

    # Use the root virtual environment instead of service-specific one
    # This ensures shared_libs are properly accessible
    root_venv_python = Path(__file__).parent.resolve() / ".venv" / "bin" / "python"
    if root_venv_python.exists():
        python_cmd = str(root_venv_python)
        if verbose:
            print(f"ℹ️  Using root virtual environment for {service_name}")
    else:
        if verbose:
            print(f"⚠️  Root virtual environment not found, using system Python")
        python_cmd = python_bin

    tests_path = service_path / "tests"
    if not tests_path.exists():
        if verbose:
            print(f"ℹ️  No tests directory found for {service_name}, skipping...")
        return 0

    # Set up environment with proper PYTHONPATH
    service_env = os.environ.copy()
    root_path = str(Path(__file__).parent.resolve())
    src_path = str(service_path / "src")
    shared_libs_path = str(Path(__file__).parent.resolve() / "shared_libs")
    
    # For cross-service imports, include all service directories
    all_service_paths = [str(Path(f"services/{s}").resolve()) for s in ['service_tracking', 'service_training', 'service_cloud'] if Path(f"services/{s}").exists()]
    
    # Construct PYTHONPATH: shared_libs first, then root directory, then src directory, then all service directories
    python_path_parts = [shared_libs_path, root_path, src_path] + all_service_paths
    service_env["PYTHONPATH"] = ":".join(python_path_parts)
    
    exit_code = 0

    # Determine which tests to run based on category
    if category == "unit" or category == "all":
        # Run unit tests (exclude integration tests)
        if verbose:
            print(f"\n🏃 Running unit tests for {service_name}")
        exit_code = run_command(
            [python_cmd, "-m", "pytest", "tests/", "-v", "-m", "not integration and not performance and not e2e", "--ignore", "tests/integration/"],
            f"Unit tests for {service_name}",
            cwd=str(service_path),  # Run from service directory
            env=service_env,
            verbose=verbose
        )

    if category == "integration" or category == "all":
        # Run integration tests
        integration_path = service_path / "tests" / "integration"
        if integration_path.exists():
            if verbose:
                print(f"\n🔗 Running integration tests for {service_name}")
            integration_exit_code = run_command(
                [python_cmd, "-m", "pytest", "tests/integration/", "-v", "-c", str(Path(__file__).parent.resolve() / "pytest.ini")],
                f"Integration tests for {service_name}",
                cwd=str(service_path),  # Run from service directory
                env=service_env,
                verbose=verbose
            )
            if integration_exit_code != 0:
                exit_code = integration_exit_code

    if category == "performance" or category == "all":
        # Run performance tests
        performance_path = service_path / "tests" / "performance"
        if performance_path.exists():
            if verbose:
                print(f"\n⚡ Running performance tests for {service_name}")
            performance_exit_code = run_command(
                [python_cmd, "-m", "pytest", "tests/performance/", "-v", "-m", "performance", "-c", str(Path(__file__).parent.resolve() / "pytest.ini")],
                f"Performance tests for {service_name}",
                cwd=str(service_path),  # Run from service directory
                env=service_env,
                verbose=verbose
            )
            if performance_exit_code != 0:
                exit_code = performance_exit_code

    if category == "e2e" or category == "all":
        # Run end-to-end tests
        e2e_path = service_path / "tests" / "e2e"
        if e2e_path.exists():
            if verbose:
                print(f"\n🎯 Running end-to-end tests for {service_name}")
            e2e_exit_code = run_command(
                [python_cmd, "-m", "pytest", "tests/e2e/", "-v", "-m", "e2e", "-c", str(Path(__file__).parent.resolve() / "pytest.ini")],
                f"End-to-end tests for {service_name}",
                cwd=str(service_path),  # Run from service directory
                env=service_env,
                verbose=verbose
            )
            if e2e_exit_code != 0:
                exit_code = e2e_exit_code

    return exit_code

def main():
    """Main test runner function."""
    args = parse_arguments()

    # Get the Python executable
    python_bin = os.environ.get('PYTHON_EXECUTABLE', sys.executable)

    all_services = ['service_tracking', 'service_training', 'service_cloud']

    # Filter services based on arguments
    if args.service:
        services = [args.service]
        print(f"🎯 Testing specific service: {args.service}")
    elif args.sv_only:
        services = []  # No service-specific tests for SV-only
        print("🔍 Testing System Verification (SV) tests only")
    else:
        services = all_services
        print("🧪 Testing all services")

    if args.sv_only:
        print("📂 Testing SV categories: integration, performance")
    elif args.category != "all":
        print(f"📂 Testing category: {args.category}")
    else:
        print("📂 Testing all categories")

    failed_tests = []

    # Prepare environment for root tests
    root_env = os.environ.copy()
    root_path = str(Path(__file__).parent.resolve())
    shared_libs_path = str(Path(__file__).parent.resolve() / "shared_libs")

    # For integration tests, include all service src directories
    service_src_paths = []
    for service in all_services:
        service_src = Path(f"services/{service}/src")
        if service_src.exists():
            service_src_paths.append(str(service_src.resolve()))

    # Construct PYTHONPATH: shared_libs first, then root directory, then all service src directories
    python_path_parts = [shared_libs_path, root_path] + service_src_paths
    root_env["PYTHONPATH"] = ":".join(python_path_parts)

    print("🧪 Starting Multi-Service Test Suite")
    print("=" * 60)

    # Determine which phases to run based on category and sv-only option
    if args.sv_only:
        run_unit = False
        run_integration = True  # SV integration tests
        run_performance = True  # SV performance tests
        run_e2e = False
    else:
        run_unit = args.category in ["unit", "all"]
        run_integration = args.category in ["integration", "all"]
        run_performance = args.category in ["performance", "all"]
        run_e2e = args.category in ["e2e", "all"]

    # Phase 1: Service unit tests (fastest)
    if run_unit:
        print(f"\n🏃 Phase 1: Service Unit Tests")
        for service in services:
            if args.verbose:
                print(f"\n📦 Testing {service}")
            exit_code = run_service_tests(service, python_bin, "unit", args.verbose)
            if exit_code != 0:
                failed_tests.append(f"{service} unit tests")

    # Phase 2: Service integration tests
    if run_integration:
        print(f"\n🔗 Phase 2: Integration Tests")
        integration_failed = False
        for service in services:
            service_path = Path(f"services/{service}")
            integration_path = service_path / "tests" / "integration"
            if integration_path.exists():
                if args.verbose:
                    print(f"\n🔗 Testing {service} integration")
                exit_code = run_command(
                    [python_bin, "-m", "pytest", f"services/{service}/tests/integration/", "-v", "-c", str(Path(__file__).parent.resolve() / "pytest.ini")],
                    f"{service} integration tests",
                    cwd=str(Path(__file__).parent.resolve()),  # Run from root directory
                    env=root_env,
                    verbose=args.verbose
                )
                if exit_code != 0:
                    failed_tests.append(f"{service} integration tests")
                    integration_failed = True

        # If no service-specific integration tests found, try root level
        if not integration_failed and Path("sv-tests/integration").exists():
            exit_code = run_command(
                [python_bin, "-m", "pytest", "sv-tests/integration/", "-v", "-c", str(Path(__file__).parent.resolve() / "pytest.ini")],
                "Cross-service integration tests",
                cwd=str(Path(__file__).parent.resolve()),  # Run from root directory
                env=root_env,
                verbose=args.verbose
            )
            if exit_code != 0:
                failed_tests.append("Integration tests")

    # Phase 3: Performance tests
    if run_performance:
        print(f"\n⚡ Phase 3: Performance Tests")
        if Path("sv-tests/performance").exists():
            exit_code = run_command(
                [python_bin, "-m", "pytest", "sv-tests/performance/", "-v", "-m", "performance"],
                "Performance tests",
                cwd=".",
                env=root_env,
                verbose=args.verbose
            )
            if exit_code != 0:
                failed_tests.append("Performance tests")

    # Phase 4: End-to-end tests (slowest)
    if run_e2e:
        print(f"\n🎯 Phase 4: End-to-End Tests")
        if Path("sv-tests").exists():
            exit_code = run_command(
                [python_bin, "-m", "pytest", "sv-tests/", "-v", "-m", "e2e"],
                "End-to-end tests",
                cwd=".",
                env=root_env,
                verbose=args.verbose
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
    print(f"\n🔗 Phase 2: Integration Tests")
    integration_failed = False
    for service in services:
        service_path = Path(f"services/{service}")
        integration_path = service_path / "tests" / "integration"
        if integration_path.exists():
            print(f"\n� Testing {service} integration")
            exit_code = run_command(
                [python_bin, "-m", "pytest", "sv-tests/integration/", "-v"],
                f"{service} integration tests",
                cwd=str(service_path),
                env=root_env
            )
            if exit_code != 0:
                failed_tests.append(f"{service} integration tests")
                integration_failed = True
    
    # If no service-specific integration tests found, try root level
    if not integration_failed and Path("sv-tests/integration").exists():
        exit_code = run_command(
            [python_bin, "-m", "pytest", "sv-tests/integration/", "-v"],
            "Cross-service integration tests",
            cwd=".",
            env=root_env
        )
        if exit_code != 0:
            failed_tests.append("Integration tests")

    # Phase 3: Performance tests
    print(f"\n⚡ Phase 3: Performance Tests")
    if Path("sv-tests/performance").exists():
        exit_code = run_command(
            [python_bin, "-m", "pytest", "sv-tests/performance/", "-v", "-m", "performance"],
            "Performance tests",
            cwd=".",
            env=root_env
        )
        if exit_code != 0:
            failed_tests.append("Performance tests")

    # Phase 4: End-to-end tests (slowest)
    print(f"\n🎯 Phase 4: End-to-End Tests")
    if Path("sv-tests").exists():
        exit_code = run_command(
            [python_bin, "-m", "pytest", "sv-tests/", "-v", "-m", "e2e"],
            "End-to-end tests",
            cwd=".",
            env=root_env
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
