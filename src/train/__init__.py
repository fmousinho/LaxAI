"""
LaxAI Training Package

This package contains all modules related to data preparation, model training,
and evaluation for the LaxAI project.
"""

from .augmentation import augment_images
from .dataprep_pipeline import DataPrepPipeline, run_dataprep_pipeline
from .dataset import LacrossePlayerDataset
from .evaluator import ModelEvaluator
from .siamesenet import SiameseNet
from .train_pipeline import TrainPipeline

__all__ = [
    # augmentation
    'augment_images',
    # dataprep_pipeline
    'DataPrepPipeline',
    'run_dataprep_pipeline',
    # dataset
    'LacrossePlayerDataset',
    # evaluator
    'ModelEvaluator',
    # siamesenet
    'SiameseNet',
    # train_pipeline
    'TrainPipeline',
]