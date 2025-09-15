"""Helper classes and dummy datasets for training-related tests.

Renamed from test_training.py to training_helpers.py to avoid pytest collecting
two different modules named test_training (integration vs unit) which caused
import file mismatch errors during test discovery.
"""

import numpy as np  # noqa: F401  (retained for potential future helpers)
import torch

from services.service_training.src.siamesenet import SiameseNet


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
        super().__init__(
            embedding_dim=emb_dim,
            use_cbam=False,
            dropout_rate=0.0
        )
