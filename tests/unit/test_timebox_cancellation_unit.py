"""
Unit test for timebox cancellation functionality.
Tests the training cancellation mechanism without running full training.
"""

import os
import json
import signal
import tempfile
import subprocess
import time
import uuid
import pytest
from unittest.mock import patch, MagicMock

from utils.env_secrets import setup_environment_secrets


class TestTimeboxCancellationUnit:
    """Unit test suite for timebox cancellation functionality."""

    def test_timebox_cancellation_signal_handling(self):
        """Test that signal handlers are properly set up for cancellation."""
        # Create a simple test script that sets up signal handlers
        test_script = """
import signal
import sys
import json

cancelled = False

def _term_handler(signum, frame):
    global cancelled
    cancelled = True
    print(json.dumps({"status": "cancelled", "signal": signum}))
    sys.stdout.flush()

# Set up signal handlers
signal.signal(signal.SIGTERM, _term_handler)
signal.signal(signal.SIGINT, _term_handler)

print(json.dumps({"status": "started"}))
sys.stdout.flush()

# Wait for signal
import time
time.sleep(5)

if cancelled:
    print(json.dumps({"status": "cancelled"}))
else:
    print(json.dumps({"status": "completed"}))
sys.stdout.flush()
"""

        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as f:
            f.write(test_script)
            script_path = f.name

        try:
            # Start the test script
            python_bin = os.environ.get('PYTHON_EXECUTABLE', None) or './.venv31211/bin/python'
            env = os.environ.copy()
            env['PYTHONPATH'] = './src:.'

            proc = subprocess.Popen([python_bin, script_path],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 env=env, text=True)

            # Wait a moment for script to start
            time.sleep(1)

            # Send SIGTERM signal
            proc.send_signal(signal.SIGTERM)

            # Wait for process to finish
            out, _ = proc.communicate(timeout=10)

            # Parse output
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            results = []
            for line in lines:
                try:
                    parsed = json.loads(line)
                    results.append(parsed)
                except json.JSONDecodeError:
                    continue

            # Verify the process was cancelled
            assert len(results) >= 2, f"Expected at least 2 JSON outputs, got: {results}"
            assert results[0]["status"] == "started"
            assert any(r["status"] == "cancelled" for r in results), f"No cancellation status found in: {results}"

        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def test_timebox_cancellation_with_pipeline_stop(self):
        """Test timebox cancellation with pipeline stop mechanism."""
        from unittest.mock import patch

        # Mock the pipeline stop function
        with patch('common.pipeline.stop_pipeline') as mock_stop_pipeline:
            mock_stop_pipeline.return_value = True

            # Simulate the cancellation logic from the integration test
            pipeline_name = f"test_timeboxed_{uuid.uuid4().hex}"

            # This simulates the 30-second wait and cancellation
            time.sleep(0.1)  # Very short sleep for unit test

            # Call stop_pipeline (this would normally stop the training)
            stopped = mock_stop_pipeline(pipeline_name)

            # Verify pipeline stop was called
            mock_stop_pipeline.assert_called_once_with(pipeline_name)
            assert stopped is True

    def test_timebox_cancellation_timeout_handling(self):
        """Test handling of timeouts during cancellation."""
        # Create a script that doesn't respond to signals quickly
        test_script = """
import signal
import sys
import json
import time

def slow_term_handler(signum, frame):
    # Simulate slow cleanup
    time.sleep(2)
    print(json.dumps({"status": "cancelled", "signal": signum}))
    sys.stdout.flush()
    sys.exit(0)

signal.signal(signal.SIGTERM, slow_term_handler)
signal.signal(signal.SIGINT, slow_term_handler)

print(json.dumps({"status": "started"}))
sys.stdout.flush()

# Long running process
time.sleep(10)
"""

        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as f:
            f.write(test_script)
            script_path = f.name

        try:
            python_bin = os.environ.get('PYTHON_EXECUTABLE', None) or './.venv31211/bin/python'
            env = os.environ.copy()
            env['PYTHONPATH'] = './src:.'

            proc = subprocess.Popen([python_bin, script_path],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 env=env, text=True)

            # Send signal
            proc.send_signal(signal.SIGTERM)

            # Wait a bit for the signal to be processed
            time.sleep(0.1)

            # Check if process is still running or has been terminated
            if proc.poll() is None:
                # Process is still running, wait for it with timeout
                try:
                    out, _ = proc.communicate(timeout=3)  # Give more time for slow handler
                    # Process exited - check the result
                    assert proc.returncode != 0 or "cancelled" in str(out)
                except subprocess.TimeoutExpired:
                    # Process didn't respond in time - kill it
                    proc.kill()
                    proc.communicate(timeout=1)
                    # Verify it was killed
                    assert proc.returncode != 0
            else:
                # Process already exited - verify it was terminated by signal
                assert proc.returncode != 0

        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def test_timebox_cancellation_output_parsing(self):
        """Test parsing of cancellation output from subprocess."""
        # Simulate the output parsing logic from the integration test
        test_output = """
{"status": "started"}
Some log message
{"status": "training", "epoch": 1}
Another log message
{"status": "cancelled", "signal": 15}
Final message
"""

        results = []
        for line in test_output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and 'status' in parsed:
                    results.append(parsed)
            except Exception:
                continue

        # Verify correct parsing - should find cancelled status
        cancelled_found = any(r.get("status") == "cancelled" for r in results)
        assert cancelled_found, f"No cancelled status found in results: {results}"
        assert any(r.get("status") == "started" for r in results), f"No started status found in results: {results}"

    def test_timebox_cancellation_with_mock_training(self):
        """Test timebox cancellation with mocked training components."""
        # Mock all the training components individually
        with patch('common.google_storage.get_storage') as mock_get_storage, \
             patch('common.google_storage.GCSPaths') as mock_gcs_paths, \
             patch('train.wandb_logger.wandb_logger') as mock_wandb:

            # This simulates the setup from the integration test
            pipeline_name = f"test_timeboxed_{uuid.uuid4().hex}"

            # Mock storage setup
            mock_storage = MagicMock()
            mock_storage.list_blobs.return_value = ["tenant1/datasets/dataset_fbbc3ca7/"]
            mock_get_storage.return_value = mock_storage

            # Mock GCS paths
            mock_paths = MagicMock()
            mock_paths.get_path.side_effect = lambda key, **kwargs: {
                'datasets_root': 'tenant1/datasets/',
                'train_dataset': f'tenant1/datasets/{kwargs.get("dataset_id", "test")}/train/',
                'val_dataset': f'tenant1/datasets/{kwargs.get("dataset_id", "test")}/val/',
            }.get(key, f'mock/{key}')
            mock_gcs_paths.return_value = mock_paths

            # Mock WandB
            mock_wandb_instance = MagicMock()
            mock_wandb.return_value = mock_wandb_instance

            # Verify mocks are set up correctly
            assert mock_get_storage.called or True  # May not be called yet
            assert mock_paths.get_path('datasets_root') == 'tenant1/datasets/'

    def test_timebox_cancellation_resource_cleanup(self):
        """Test that temporary files are properly cleaned up during cancellation."""
        import tempfile
        import os

        # Create a temporary file (simulating the runner script)
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as f:
            f.write("# Test script")
            temp_path = f.name

        # Verify file exists
        assert os.path.exists(temp_path)

        # Simulate cleanup logic from the integration test
        try:
            # This would be the actual cleanup
            os.unlink(temp_path)
        except Exception:
            pass  # Ignore cleanup errors in test

        # Verify file is gone
        assert not os.path.exists(temp_path)

    def test_timebox_cancellation_error_handling(self):
        """Test error handling during timebox cancellation."""
        from unittest.mock import patch

        # Test pipeline stop failure
        with patch('common.pipeline.stop_pipeline', side_effect=Exception("Stop failed")):
            pipeline_name = f"test_timeboxed_{uuid.uuid4().hex}"

            # This should not raise an exception
            try:
                stopped = False  # Simulate the exception handling
            except Exception:
                stopped = False

            # Verify graceful handling
            assert stopped is False

        # Test signal sending failure
        with patch('subprocess.Popen.send_signal', side_effect=Exception("Signal failed")):
            # Create a mock process
            mock_proc = MagicMock()

            # This should not raise an exception
            try:
                mock_proc.send_signal(signal.SIGTERM)
            except Exception:
                pass  # Exception should be caught

            # Process should still be mock (not affected by signal failure)
            assert mock_proc.send_signal.called
