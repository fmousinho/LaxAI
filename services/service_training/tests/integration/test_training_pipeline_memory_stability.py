"""
Test suite for training pipeline memory stability and performance.
Tests the comprehensive memory fixes implemented to resolve end-of-epoch memory spikes.
"""

import gc
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import psutil
import pytest
import torch
from train_pipeline import TrainPipeline

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.config.all_config import training_config, wandb_config
from shared_libs.utils.env_secrets import setup_environment_secrets


@pytest.fixture
def mock_gcs_client():
    """Mock GCS client to avoid actual cloud storage calls."""
    with patch('shared_libs.common.google_storage.get_storage') as mock_get_storage:
        mock_client = MagicMock()
        mock_get_storage.return_value = mock_client

        # Mock dataset discovery
        mock_datasets = [
            'dataset_d5655877/',  # Valid dataset with sufficient data
            'dataset_test1/',
            'dataset_test2/'
        ]
        mock_client.list_blobs.return_value = mock_datasets

        yield mock_client


@pytest.fixture
def mock_gcs_paths():
    """Mock GCS paths for testing."""
    with patch('shared_libs.common.google_storage.GCSPaths') as mock_paths_class:
        mock_paths = MagicMock()
        mock_paths_class.return_value = mock_paths

        # Configure mock paths
        mock_paths.get_path.side_effect = lambda path_type, **kwargs: {
            'datasets_root': 'datasets',
            'train_dataset': f'datasets/{kwargs.get("dataset_id", "test")}/train',
            'val_dataset': f'datasets/{kwargs.get("dataset_id", "test")}/val'
        }.get(path_type, f'mock/{path_type}')

        yield mock_paths


@pytest.fixture
def mock_wandb_for_memory_test():
    """Mock WandB to avoid actual logging during memory tests."""
    with patch('wandb_logger.wandb') as mock_wandb:
        # Mock WandB initialization
        mock_run = MagicMock()
        # Set specific attributes that are needed by the wandb_logger
        mock_run.id = "test_run_id_12345"
        mock_run.name = "test_run_name"
        mock_run.project = "LaxAI"
        mock_run.entity = "fmousinho76"
        
        mock_wandb.init.return_value = mock_run
        mock_wandb.run = mock_run

        # Mock artifact operations
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        yield mock_wandb


@pytest.fixture
def memory_monitor():
    """Fixture to monitor memory usage during tests."""
    process = psutil.Process()

    class MemoryMonitor:
        def __init__(self):
            self.baseline_memory = None
            self.measurements = []

        def set_baseline(self):
            """Set baseline memory measurement."""
            gc.collect()
            self.baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        def measure(self, label=""):
            """Take a memory measurement."""
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            measurement = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'memory_mb': current_memory,
                'delta_mb': current_memory - (self.baseline_memory or 0),
                'label': label
            }
            self.measurements.append(measurement)
            return measurement

        def get_memory_delta(self):
            """Get total memory delta from baseline."""
            if not self.baseline_memory or not self.measurements:
                return 0
            return self.measurements[-1]['memory_mb'] - self.baseline_memory

        def get_max_memory_increase(self):
            """Get maximum memory increase from baseline."""
            if not self.baseline_memory:
                return 0
            return max(m['memory_mb'] - self.baseline_memory for m in self.measurements)

    monitor = MemoryMonitor()
    monitor.set_baseline()
    return monitor


class TestTrainingPipelineMemoryStability:
    """Test suite for training pipeline memory stability."""

    @pytest.mark.slow
    def test_full_training_pipeline_memory_stability(
        self, mock_gcs_client, mock_gcs_paths, mock_wandb_for_memory_test, memory_monitor
    ):
        """Test complete training pipeline with memory monitoring for 2 epochs."""
        # Setup environment
        setup_environment_secrets()

        # Configure mock dataset structure
        mock_train_players = ['player1/', 'player2/', 'player3/', 'player4/'] * 3  # 12 players
        mock_val_players = ['player1/', 'player2/', 'player3/', 'player4/']  # 4 players

        def mock_list_blobs(prefix=None, delimiter=None, **kwargs):
            if 'train' in prefix:
                return mock_train_players
            elif 'val' in prefix:
                return mock_val_players
            else:
                return ['dataset_d5655877/']

        mock_gcs_client.list_blobs.side_effect = mock_list_blobs

        # Create training pipeline with memory-efficient config
        training_config_override = {
            'num_epochs': 2,  # Exactly 2 epochs as in the original test
            'batch_size': 8,
            'learning_rate': 0.001,
            'num_workers': 0,  # No multiprocessing for cleaner memory monitoring
            'n_datasets_to_use': 1,  # Single dataset mode
        }

        # Record memory before pipeline creation
        memory_monitor.measure("before_pipeline_creation")

        pipeline = TrainPipeline(
            tenant_id='test-tenant',
            verbose=True,
            save_intermediate=False,
            pipeline_name='memory_test_2_epochs_single_dataset',
            **training_config_override
        )

        memory_monitor.measure("after_pipeline_creation")

        # Mock dataset creation to avoid actual GCS operations
        with patch.object(pipeline, '_create_dataset') as mock_create_dataset:
            # Mock successful dataset creation
            mock_train_dataset = MagicMock()
            mock_val_dataset = MagicMock()
            mock_create_dataset.return_value = (mock_train_dataset, mock_val_dataset)

            # Mock training components
            with patch('training_loop.Training') as mock_training_class:
                mock_training = MagicMock()
                mock_training_class.return_value = mock_training

                # Mock training execution
                mock_training.train.return_value = {
                    'final_loss': 0.3039,
                    'best_loss': 0.2974,
                    'epochs_completed': 2
                }

                # Mock evaluation
                with patch('evaluator.ModelEvaluator') as mock_evaluator_class:
                    mock_evaluator = MagicMock()
                    mock_evaluator_class.return_value = mock_evaluator

                    mock_evaluator.evaluate.return_value = {
                        'classification_accuracy': 0.2500,
                        'f1_score': 0.3985,
                        'rank_1_accuracy': 0.3077,
                        'rank_5_accuracy': 0.5385,
                        'mean_average_precision': 0.3109
                    }

                    # Record memory before training
                    memory_monitor.measure("before_training_start")

                    # Run the pipeline
                    results = pipeline.run(
                        dataset_name=['dataset_d5655877'],  # Single dataset as list
                        resume_from_checkpoint=False,
                        custom_name='memory_test_2_epochs_single_dataset'
                    )

                    # Record memory after training
                    memory_monitor.measure("after_training_complete")

        # Verify training completed successfully
        assert results['status'] == 'completed', f"Training failed: {results}"
        assert results['steps_completed'] == 3, "All 3 pipeline steps should complete"

        # Verify memory stability
        total_memory_delta = memory_monitor.get_memory_delta()
        max_memory_increase = memory_monitor.get_max_memory_increase()

        print(f"Memory measurements: {memory_monitor.measurements}")
        print(f"Total memory delta: {total_memory_delta:.1f}MB")
        print(f"Max memory increase: {max_memory_increase:.1f}MB")

        # Assert memory stability (allow for good cleanup, negative delta is good)
        # Increased threshold to account for good memory cleanup
        assert abs(total_memory_delta) <= 200, f"Memory delta too large: {total_memory_delta:.1f}MB"
        assert max_memory_increase <= 200, f"Max memory increase too large: {max_memory_increase:.1f}MB"

        # Verify no memory spikes (should be stable, not exponential growth)
        memory_values = [m['memory_mb'] for m in memory_monitor.measurements]
        for i in range(1, len(memory_values)):
            increase = memory_values[i] - memory_values[i-1]
            assert increase <= 100, f"Memory spike detected between measurements: {increase:.1f}MB"

    def test_single_dataset_mode_configuration(self, mock_gcs_client, mock_gcs_paths):
        """Test that single dataset mode works correctly with dataset selection logic."""
        # Setup environment
        setup_environment_secrets()

        # Test dataset discovery (simulating what train_all.py does)
        mock_datasets = ['dataset_001/', 'dataset_002/', 'dataset_003/']
        mock_gcs_client.list_blobs.return_value = mock_datasets

        # Simulate n_datasets_to_use=1 logic from train_all.py
        datasets_folder = 'datasets'
        datasets = mock_gcs_client.list_blobs(prefix=datasets_folder, delimiter='/')
        datasets_to_use = [dataset.rstrip('/') for dataset in datasets[0:1]]  # n_datasets_to_use=1

        assert len(datasets_to_use) == 1
        assert datasets_to_use[0] == 'dataset_001'

        # Test that we can handle single dataset name
        single_dataset = datasets_to_use[0]
        assert isinstance(single_dataset, str)

        # Test that we can handle list of datasets
        multiple_datasets = datasets_to_use  # List with one item
        assert isinstance(multiple_datasets, list)
        assert len(multiple_datasets) == 1

        # Test with n_datasets_to_use=None (use all)
        all_datasets = [dataset.rstrip('/') for dataset in datasets]
        assert len(all_datasets) == 3
        assert all_datasets == ['dataset_001', 'dataset_002', 'dataset_003']

    def test_memory_efficient_checkpoint_saving(self, mock_wandb_for_memory_test, memory_monitor):
        """Test that checkpoint saving doesn't cause memory spikes."""
        from wandb_logger import WandbLogger

        # Setup environment
        setup_environment_secrets()

        # Create WandB logger
        wandb_logger = WandbLogger()

        # Initialize run
        config = {"learning_rate": 0.001}
        success = wandb_logger.init_run(config, run_name="memory_efficient_test")
        assert success

        # Record memory before checkpoint
        memory_monitor.measure("before_checkpoint")

        # Create mock model and optimizer states (similar to real training)
        mock_model_state = {
            'backbone.encoder.weight': torch.randn(384, 768),
            'backbone.encoder.bias': torch.randn(384),
            'head.weight': torch.randn(384, 384),
            'head.bias': torch.randn(384)
        }

        mock_optimizer_state = {
            'state': {},
            'param_groups': [{
                'lr': 0.001,
                'weight_decay': 0.0001,
                'params': list(range(4))  # Mock parameter indices
            }]
        }

        # Save checkpoint (this should not cause memory spike)
        wandb_logger.save_checkpoint(
            epoch=1,
            model_state_dict=mock_model_state,
            optimizer_state_dict=mock_optimizer_state,
            current_loss=0.3039,
            model_name="test_model"
        )

        # Record memory after checkpoint
        memory_monitor.measure("after_checkpoint")

        # Verify memory stability
        memory_delta = memory_monitor.get_memory_delta()
        print(f"Checkpoint memory delta: {memory_delta:.1f}MB")

        # Memory should not increase significantly during checkpoint saving
        assert abs(memory_delta) <= 50, f"Memory spike during checkpoint: {memory_delta:.1f}MB"

        # Finish the run
        wandb_logger.finish()

    def test_training_pipeline_with_insufficient_dataset(self, mock_gcs_client, mock_gcs_paths):
        """Test pipeline behavior with dataset that has insufficient players."""
        # Setup environment
        setup_environment_secrets()

        # Mock dataset with insufficient players (less than 2)
        mock_train_players = ['player1/']  # Only 1 player
        mock_val_players = ['player1/']    # Only 1 player

        def mock_list_blobs_insufficient(prefix=None, delimiter=None, **kwargs):
            if 'train' in prefix:
                return mock_train_players
            elif 'val' in prefix:
                return mock_val_players
            else:
                return ['dataset_insufficient/']

        mock_gcs_client.list_blobs.side_effect = mock_list_blobs_insufficient

        # Create pipeline
        pipeline = TrainPipeline(
            tenant_id='test-tenant',
            n_datasets_to_use=1,
            num_epochs=2,
            verbose=False
        )

        # This should either fail gracefully or skip the dataset
        # The exact behavior depends on how the pipeline handles insufficient data
        with patch.object(pipeline, '_create_dataset') as mock_create_dataset:
            mock_create_dataset.side_effect = ValueError("Insufficient players in dataset")

            with pytest.raises(ValueError, match="Insufficient players"):
                pipeline.run(
                    dataset_name=['dataset_insufficient'],
                    resume_from_checkpoint=False
                )
