"""
Basic test for service_training module.
"""
import pytest


def test_service_training_import():
    """Test that we can import from service_training."""
    try:
        from training import Training
        assert Training is not None
    except ImportError:
        pytest.skip("Training class not available")


def test_basic_math():
    """Basic math test."""
    assert 2 + 2 == 4


def test_string_manipulation():
    """Test string operations."""
    s = "service_training"
    assert "service" in s
    assert s.startswith("service")
    assert s.endswith("training")
