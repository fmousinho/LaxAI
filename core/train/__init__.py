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
    run_dataprep_pipeline
)

from .siamesenet import SiameseNet

from .training import Training

from .evaluator import ModelEvaluator

from .wandb_logger import wandb_logger

__all__ = [
    'augment_images',
    'test_augmentation',
    'DataPrepPipeline',
    'run_dataprep_pipeline',
    'SiameseNet',
    'Training',
    'ModelEvaluator',
    'wandb_logger'
]