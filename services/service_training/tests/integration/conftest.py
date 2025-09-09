"""
Integration tests for service_training.

These tests verify components working together:
- Test complete training pipelines with shared_libs integration
- Test WandB logging with shared_libs components
- Test data flow between components
- May require shared_libs but NOT other services
- Can be slower than unit tests but faster than performance tests
"""

import os
import sys
from pathlib import Path

import pytest

# Add src and shared_libs to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'service_training' / 'src'))
sys.path.insert(0, str(project_root / 'shared_libs'))

# Import shared_libs components for integration testing
try:
    from shared_libs.utils.env_secrets import setup_environment_secrets
    SHARED_LIBS_AVAILABLE = True
except ImportError:
    SHARED_LIBS_AVAILABLE = False

@pytest.fixture(scope="session", autouse=True)
def setup_integration_test_environment():
    """Set up environment for integration tests."""
    if SHARED_LIBS_AVAILABLE:
        try:
            setup_environment_secrets()
        except Exception:
            # Allow tests to handle missing secrets gracefully
            pass

@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return {
        'tenant_id': 'test-tenant',
        'verbose': True,
        'custom_name': 'integration-test',
        'resume_from_checkpoint': False,
        'wandb_tags': ['integration', 'test'],
        'training_params': {
            'batch_size': 4,
            'learning_rate': 0.001,
            'num_epochs': 1
        }
    }

@pytest.fixture
def shared_libs_available():
    """Check if shared_libs are available for integration testing."""
    return SHARED_LIBS_AVAILABLE
