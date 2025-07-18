"""
LaxAI Core Training Package

This package contains the core training modules for the LaxAI lacrosse video analysis system.
"""

from .augmentation import (
    augment_images,
    test_augmentation
)

from .train_pipeline import (
    TrainPipeline,
    run_training_pipeline
)

__all__ = [
    'augment_images',
    'test_augmentation',
    'TrainPipeline',
    'run_training_pipeline'
]