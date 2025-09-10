"""
Simple test to verify VS Code testing is working.
"""
import pytest


def test_simple_assertion():
    """A simple test that should always pass."""
    assert 1 + 1 == 2


def test_string_operations():
    """Test basic string operations."""
    test_string = "hello world"
    assert "hello" in test_string
    assert test_string.upper() == "HELLO WORLD"
    assert len(test_string) == 11




def test_environment_setup():
    """Test that basic environment is working."""
    import sys
    assert sys.version_info.major >= 3
    assert "python" in sys.executable.lower()
