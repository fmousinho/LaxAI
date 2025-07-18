"""
LaxAI Core Training Package

This package contains the core training modules for the LaxAI lacrosse video analysis system.
"""

from .augmentation import (
    augment_images,
    test_augmentation
)

from .dataprep_pipeline import (
    DataPrepPipeline,
    run_training_pipeline,
    run_dataprep_pipeline
)

__all__ = [
    'augment_images',
    'test_augmentation',
    'DataPrepPipeline',
    'run_training_pipeline',
    'run_dataprep_pipeline'
]