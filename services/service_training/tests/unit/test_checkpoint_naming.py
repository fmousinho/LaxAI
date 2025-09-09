#!/usr/bin/env python3
"""
Unit tests for checkpoint naming logic.
"""
import os
import sys

import pytest


def test_checkpoint_naming_no_run():
    """Test checkpoint naming when no run is active."""
    try:
        from wandb_logger import WandbLogger

        wandb_logger = WandbLogger()
        wandb_logger.run = None
        checkpoint_name = wandb_logger._get_checkpoint_name()

        assert checkpoint_name is not None
        assert isinstance(checkpoint_name, str)
        print(f'✅ Default checkpoint name (no run): {checkpoint_name}')

    except Exception as e:
        pytest.fail(f"Could not test checkpoint naming: {e}")


def test_checkpoint_naming_with_run():
    """Test checkpoint naming with an active run."""
    try:
        from wandb_logger import WandbLogger

        wandb_logger = WandbLogger()

        # Mock a run
        class MockRun:
            def __init__(self, name):
                self.name = name

        wandb_logger.run = MockRun('my-training-run-v1')
        checkpoint_name = wandb_logger._get_checkpoint_name()

        assert checkpoint_name is not None
        assert 'my-training-run-v1' in checkpoint_name
        print(f'✅ Checkpoint name with run: {checkpoint_name}')

    except Exception as e:
        pytest.fail(f"Could not test checkpoint naming with run: {e}")


def test_checkpoint_naming_sanitization():
    """Test checkpoint naming with special characters that need sanitization."""
    try:
        from wandb_logger import WandbLogger

        wandb_logger = WandbLogger()

        # Mock a run with special characters
        class MockRun:
            def __init__(self, name):
                self.name = name

        wandb_logger.run = MockRun('my@special#run!2024')
        checkpoint_name = wandb_logger._get_checkpoint_name()

        assert checkpoint_name is not None
        assert '@' not in checkpoint_name
        assert '#' not in checkpoint_name
        assert '!' not in checkpoint_name
        print(f'✅ Checkpoint name with sanitized run: {checkpoint_name}')

    except Exception as e:
        pytest.fail(f"Could not test checkpoint naming sanitization: {e}")


def test_checkpoint_naming_test_prefix():
    """Test checkpoint naming for test runs."""
    try:
        from wandb_logger import WandbLogger

        wandb_logger = WandbLogger()

        # Mock a test run
        class MockRun:
            def __init__(self, name):
                self.name = name

        wandb_logger.run = MockRun('test-run-123')
        checkpoint_name = wandb_logger._get_checkpoint_name()

        assert checkpoint_name is not None
        assert 'test-run-123' in checkpoint_name
        print(f'✅ Test checkpoint name: {checkpoint_name}')

    except Exception as e:
        pytest.fail(f"Could not test checkpoint naming for test runs: {e}")
