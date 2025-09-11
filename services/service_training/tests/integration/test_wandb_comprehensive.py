"""
Comprehensive WandB Logger Test Suite
=====================================

This module consolidates all WandB logger tests into a single, optimized test suite
that covers all functionality without redundancy.

Test Categories:
- Core Functionality: Initialization, basic operations, naming
- Async Operations: Async uploads, memory monitoring, cleanup
- Artifact Management: Checkpoints, models, versioning, cleanup
- Configuration: Retention policies, sanitization, test detection
- Integration: Real WandB API interaction with proper cleanup
"""

import gc
import importlib.util
import os
import time
import uuid
from typing import Any, Dict, Optional

import pytest
import torch

# Load environment secrets before running tests
try:
    from shared_libs.utils.env_secrets import setup_environment_secrets
    setup_environment_secrets()
except Exception as e:
    print(f"Warning: Could not load environment secrets: {e}")


def load_wandb_logger_module():
    """Load the WandB logger module dynamically."""
    path = os.path.join('services', 'service_training', 'src', 'wandb_logger.py')
    spec = importlib.util.spec_from_file_location('wandb_logger_mod', path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wandb_logger_module():
    """Fixture to load WandB logger module once per test module."""
    return load_wandb_logger_module()


@pytest.fixture
def logger_instance(wandb_logger_module):
    """Fixture to create a fresh logger instance for each test."""
    return wandb_logger_module.WandbLogger(enabled=True)


def requires_wandb_api():
    """Decorator to skip tests when WandB API is not available."""
    return pytest.mark.skipif(
        'WANDB_API_KEY' not in os.environ,
        reason='WANDB_API_KEY not set; skipping WandB API tests'
    )


def requires_wandb_package():
    """Decorator to skip tests when wandb package is not installed."""
    try:
        import wandb  # noqa: F401
        return lambda func: func
    except ImportError:
        return pytest.skip('wandb not installed; skipping wandb tests')


@pytest.mark.integration
class TestWandbLoggerCore:
    """Core functionality tests for WandB logger."""

    @requires_wandb_api()
    @requires_wandb_package()
    def test_initialization_and_naming(self, logger_instance):
        """Test basic initialization and smart naming."""
        run_name = f"test_core_{uuid.uuid4().hex[:8]}"
        
        # Test initialization
        assert logger_instance.init_run({'test': True}, run_name=run_name)
        assert logger_instance.initialized
        assert logger_instance._is_test_run()
        
        # Test smart naming
        checkpoint_name = logger_instance._get_artifact_name("checkpoint")
        assert checkpoint_name == "test-checkpoint"
        
        model_name = logger_instance._get_artifact_name("player_model")
        assert model_name == "test-player_model"
        
        # Test sanitization
        dirty_name = "test/with\\invalid:chars*"
        clean_name = logger_instance._sanitize_artifact_name(dirty_name)
        assert "/" not in clean_name
        assert "\\" not in clean_name
        assert ":" not in clean_name
        assert "*" not in clean_name
        
        logger_instance.finish()

    @requires_wandb_api()
    @requires_wandb_package()
    def test_configuration_and_attributes(self, logger_instance):
        """Test configuration attributes and defaults."""
        # Test default configuration
        assert hasattr(logger_instance, 'model_retention_count')
        assert hasattr(logger_instance, 'checkpoint_retention_count')
        assert logger_instance.model_retention_count == 3
        assert logger_instance.checkpoint_retention_count == 1
        
        # Test ThreadPoolExecutor initialization after init_run
        run_name = f"test_config_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({'test': True}, run_name=run_name)
        
        assert logger_instance._executor is not None
        assert logger_instance._pending_futures == []
        
        logger_instance.finish()


@pytest.mark.integration
class TestWandbLoggerAsync:
    """Async operations and memory monitoring tests."""

    @requires_wandb_api()
    @requires_wandb_package()
    def test_async_checkpoint_operations(self, logger_instance):
        """Test async checkpoint uploads and cleanup."""
        run_name = f"test_async_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({'test': True}, run_name=run_name)
        
        # Create test model
        model = torch.nn.Linear(10, 2)
        
        # Test multiple async checkpoint saves
        for epoch in range(3):
            result = logger_instance.save_checkpoint(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict={},
                loss=1.0/(epoch+1)
            )
            assert result is not None
        
        # Test async operations are queued
        assert len(logger_instance._pending_futures) > 0
        
        # Wait for completion
        logger_instance._wait_for_pending_operations()
        assert len(logger_instance._pending_futures) == 0
        
        logger_instance.finish()

    @requires_wandb_api()
    @requires_wandb_package()
    def test_memory_monitoring(self, logger_instance):
        """Test memory monitoring decorators."""
        # Check that memory monitoring decorators are applied
        assert hasattr(logger_instance.init_run, '__wrapped__')
        assert hasattr(logger_instance.save_checkpoint, '__wrapped__')
        assert hasattr(logger_instance.save_model_to_registry, '__wrapped__')
        
        run_name = f"test_memory_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({'test': True}, run_name=run_name)
        logger_instance.finish()


@pytest.mark.integration  
class TestWandbLoggerArtifacts:
    """Artifact management tests including cleanup."""

    @requires_wandb_api()
    @requires_wandb_package()
    def test_checkpoint_lifecycle(self, logger_instance):
        """Test complete checkpoint lifecycle with cleanup."""
        run_name = f"test_checkpoint_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({'test': True}, run_name=run_name)
        
        model = torch.nn.Linear(5, 1)
        
        # Save checkpoint
        artifact = logger_instance.save_checkpoint(
            epoch=1,
            model_state_dict=model.state_dict(),
            optimizer_state_dict={},
            loss=0.5
        )
        assert artifact is not None
        
        # Wait for upload
        logger_instance._wait_for_pending_operations()
        
        # Test finish with auto-cleanup
        logger_instance.finish()

    @requires_wandb_api()
    @requires_wandb_package()
    def test_model_registry_lifecycle(self, logger_instance):
        """Test complete model registry lifecycle with cleanup."""
        run_name = f"test_model_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({'test': True}, run_name=run_name)
        
        model = torch.nn.Linear(8, 2)
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        
        # Save model to registry
        logger_instance.save_model_to_registry(
            model, 
            collection_name=collection_name,
            alias='test',
            metadata={'test': True, 'version': '1.0'}
        )
        
        # Wait for upload and cleanup
        logger_instance._wait_for_pending_operations()
        
        # Test finish with auto-cleanup
        logger_instance.finish()

    @requires_wandb_api()
    @requires_wandb_package()
    def test_artifact_cleanup_functionality(self, logger_instance):
        """Test artifact cleanup methods."""
        run_name = f"test_cleanup_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({'test': True}, run_name=run_name)
        
        model = torch.nn.Linear(3, 1)
        
        # Create artifacts to cleanup
        logger_instance.save_checkpoint(
            epoch=1, 
            model_state_dict=model.state_dict(),
            optimizer_state_dict={},
            loss=0.1
        )
        
        collection_name = f"cleanup_test_{uuid.uuid4().hex[:6]}"
        logger_instance.save_model_to_registry(
            model,
            collection_name=collection_name,
            alias='test'
        )
        
        # Wait for operations
        logger_instance._wait_for_pending_operations()
        
        # Test manual cleanup (internal method)
        checkpoint_name = logger_instance._get_artifact_name("checkpoint")
        try:
            logger_instance._lightweight_artifact_delete(checkpoint_name, "model_checkpoint")
        except Exception:
            # May not exist or already deleted - acceptable
            pass
        
        model_name = logger_instance._get_artifact_name(collection_name)
        try:
            logger_instance._lightweight_artifact_delete(model_name, "model")
        except Exception:
            # May not exist or already deleted - acceptable
            pass
        
        logger_instance.finish()


@pytest.mark.integration
class TestWandbLoggerIntegration:
    """Integration tests with real WandB API."""

    @requires_wandb_api()
    @requires_wandb_package()
    def test_complete_training_simulation(self, logger_instance):
        """Simulate a complete training run with all features."""
        run_name = f"test_training_sim_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({
            'test': True,
            'epochs': 3,
            'model_type': 'test_linear'
        }, run_name=run_name)
        
        model = torch.nn.Linear(10, 3)
        optimizer_state = {'lr': 0.001, 'momentum': 0.9}
        
        # Simulate training epochs
        for epoch in range(3):
            # Log training metrics
            logger_instance.log_training_metrics(
                epoch=epoch,
                train_loss=1.0/(epoch+1),
                val_loss=0.8/(epoch+1),
                learning_rate=0.001 * (0.9 ** epoch)
            )
            
            # Save checkpoint
            logger_instance.save_checkpoint(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer_state,
                loss=1.0/(epoch+1)
            )
        
        # Save final model
        logger_instance.save_model_to_registry(
            model,
            collection_name="final_test_model",
            alias='final',
            metadata={
                'epochs_trained': 3,
                'final_loss': 1.0/3,
                'model_size': sum(p.numel() for p in model.parameters())
            }
        )
        
        # Wait for all operations
        logger_instance._wait_for_pending_operations()
        
        # Finish with auto-cleanup
        logger_instance.finish()

    @requires_wandb_api()
    @requires_wandb_package()
    def test_error_recovery_and_robustness(self, logger_instance):
        """Test error handling and recovery."""
        run_name = f"test_robustness_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({'test': True}, run_name=run_name)
        
        # Test with invalid artifact names
        try:
            logger_instance._lightweight_artifact_delete("nonexistent_artifact", "model")
        except Exception:
            # Expected to fail gracefully
            pass
        
        # Test cleanup with no artifacts
        try:
            logger_instance._cleanup_test_artifacts()
        except Exception as e:
            pytest.fail(f"Cleanup should not fail even with no artifacts: {e}")
        
        logger_instance.finish()


# Performance monitoring test (kept separate for clarity)
@pytest.mark.performance
class TestWandbLoggerPerformance:
    """Performance and memory tests."""

    @requires_wandb_api()
    @requires_wandb_package()
    def test_memory_leak_prevention(self, logger_instance):
        """Test that logger doesn't leak memory over multiple operations."""
        initial_objects = len(gc.get_objects())
        
        run_name = f"test_memory_leak_{uuid.uuid4().hex[:8]}"
        logger_instance.init_run({'test': True}, run_name=run_name)
        
        model = torch.nn.Linear(50, 10)
        
        # Perform multiple operations
        for i in range(5):
            logger_instance.save_checkpoint(
                epoch=i,
                model_state_dict=model.state_dict(),
                optimizer_state_dict={},
                loss=1.0/(i+1)
            )
        
        logger_instance._wait_for_pending_operations()
        logger_instance.finish()
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Allow some growth but flag excessive memory retention
        object_growth = final_objects - initial_objects
        assert object_growth < 3000, f"Potential memory leak: {object_growth} new objects retained"


if __name__ == "__main__":
    # Allow running this file directly for debugging
    pytest.main([__file__, "-v"])
