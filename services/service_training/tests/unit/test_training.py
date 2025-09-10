import numpy as np
import pytest
import torch
from src.siamesenet import SiameseNet
from training_loop import Training


class DummyTripletDataset(torch.utils.data.Dataset):
    def __init__(self, n=4, dim=4):
        self.n = n
        self.dim = dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return (anchor, positive, negative, label) - 4D tensors for SiameseNet
        # Create fake 3-channel images of size 32x32
        x = torch.randn(3, 32, 32)
        label = idx % 2
        return x, x, x, label


class DummyEvalDataset(torch.utils.data.Dataset):
    def __init__(self, n=4, dim=4):
        self.n = n
        self.dim = dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Create fake 3-channel images of size 32x32 for SiameseNet
        x = torch.randn(3, 32, 32)
        label = idx % 2
        # Return same shape as triplet dataset: (anchor, positive, negative, label)
        return x, x, x, label


class DummyModel(SiameseNet):
    """Wrapper around SiameseNet for testing with simplified parameters."""
    
    def __init__(self, dim=4, emb_dim=2):
        # Initialize with test-friendly parameters
        # Force consistent architecture for testing
        super().__init__(
            embedding_dim=emb_dim,
            use_cbam=False,  # Disable CBAM for faster testing
            dropout_rate=0.0
        )


def test_training_runs_and_triggers_evaluation(monkeypatch):
    # Arrange
    import evaluator as evaluator_mod
    import training_loop as training_mod

    # Force validation to run every epoch
    monkeypatch.setattr(training_mod, 'EPOCHS_PER_VAL', 1)

    # Spy on the evaluation method to ensure it's called during validation
    called = {'count': 0}
    original_evaluate = training_mod.Training._evaluate_reid_metrics

    def mock_evaluate_reid_metrics(self, dataloader):
        called['count'] += 1
        # Return a small, valid metrics dict
        return {
            'ranking_metrics': {'mean_average_precision': 0.5, 'rank_1_accuracy': 1.0, 'rank_5_accuracy': 1.0, 'rank_10_accuracy': 1.0},
            'classification_metrics': {'accuracy': 1.0},
            'distance_metrics': {'avg_distance_same_player': 0.1}
        }

    monkeypatch.setattr(training_mod.Training, '_evaluate_reid_metrics', mock_evaluate_reid_metrics)

    # Prevent wandb logger from raising during tests when not initialized
    from wandb_logger import wandb_logger as tw

    # Stub common wandb interactions used during training so tests remain isolated
    monkeypatch.setattr(tw, 'log_metrics', lambda *a, **k: None)
    monkeypatch.setattr(tw, 'save_checkpoint', lambda *a, **k: None)
    monkeypatch.setattr(tw, 'save_model_to_registry', lambda *a, **k: None)

    # Create training instance and set required hyperparameters directly
    t = Training(device=torch.device('cpu'))
    # Minimal hyperparameters required by setup_model and train
    t.num_epochs = 1
    t.batch_size = 2
    t.learning_rate = 1e-3
    t.margin = 1.0
    t.weight_decay = 0.0
    t.lr_scheduler_factor = 0.1
    t.scheduler_patience = 10
    t.scheduler_threshold = 0.01
    t.lr_scheduler_min_lr = 1e-6
    t.force_pretraining = True
    t.num_workers = 0
    t.margin_decay_rate = 1.0
    t.margin_change_threshold = 1e-6

    # Prevent loading any existing checkpoints in test to avoid architecture mismatch
    monkeypatch.setattr(tw, 'load_checkpoint', lambda *a, **k: None)
    monkeypatch.setattr(tw, 'resume_training_from_checkpoint', lambda *a, **k: 1)

    # Setup small datasets
    train_ds = DummyTripletDataset(n=2, dim=4)
    val_ds = DummyEvalDataset(n=2, dim=4)

    # Act
    t.setup_training_pipeline(DummyModel, train_ds, model_name='dummy', val_dataset=val_ds)

    # Run training for a single epoch which should trigger validation and evaluator
    model = t.train(start_epoch=1)

    # Assert
    assert isinstance(model, DummyModel)
    assert called['count'] >= 1
