"""
Test fixtures and helpers for service_training tests.
"""

import os
from typing import Any, Dict

import pytest
import torch


class WandbTestManager:
    """Simple test manager for WandB artifacts."""

    def __init__(self):
        self.tracked_artifacts = []

    def track_artifact(self, artifact_name: str):
        """Track an artifact for cleanup."""
        self.tracked_artifacts.append(artifact_name)

    def cleanup_artifacts(self):
        """Clean up tracked artifacts."""
        # In a real implementation, this would delete artifacts from WandB
        self.tracked_artifacts.clear()


@pytest.fixture(scope="function")
def wandb_test_manager():
    """Fixture providing a WandB test manager."""
    manager = WandbTestManager()
    yield manager
    # Cleanup after test
    manager.cleanup_artifacts()


def create_minimal_checkpoint(epoch: int) -> Dict[str, Any]:
    """Create a minimal checkpoint dictionary for testing."""
    # Create minimal model state
    model_state = {
        'layer.weight': torch.randn(10, 5),
        'layer.bias': torch.randn(10)
    }

    # Create minimal optimizer state
    optimizer_state = {
        'state': {},
        'param_groups': [{'lr': 0.001, 'weight_decay': 0.0}]
    }

    return {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': 1.0 - epoch * 0.1,
        'config': {'learning_rate': 0.001}
    }
