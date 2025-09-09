"""
Unit tests for service_training.

These tests focus on individual components in isolation:
- Test training loop components with mock/dummy data
- Test utility functions and helpers
- Test model components without external dependencies
- Fast execution, no external service calls
"""

import os
import sys

import pytest


# Common fixtures for unit tests
@pytest.fixture
def dummy_config():
    """Provide a dummy configuration for unit tests."""
    return {
        'batch_size': 4,
        'learning_rate': 0.001,
        'num_epochs': 1,
        'embedding_dim': 128
    }

@pytest.fixture
def mock_training_params():
    """Mock training parameters for unit tests."""
    return {
        'batch_size': 8,
        'learning_rate': 0.01,
        'num_epochs': 2,
        'optimizer': 'adam'
    }
