import os
import logging
from typing import Any, Dict, List, Optional


from core.common.pipeline_step import PipelineStep, StepStatus
from core.common.google_storage import GoogleStorageClient
from core.common.pipeline import Pipeline, PipelineStatus
from core.train.dataset import LacrossePlayerDataset
from config.transforms import get_transforms


logger = logging.getLogger(__name__)


class TrainPipeline(Pipeline):

    def __init__(self, storage_client: GoogleStorageClient):
        self.storage_client = storage_client

        step_definitions = {
            "create_dataset": {
                "description": "Create dataset from crops",
                "function": self._create_dataset
            },

        }

    def run(self, dataset_path: str) -> Dict[str, Any]:
        # Implement the training pipeline logic here
        pass


    def _create_dataset(self, context: dict) -> dict:
        dataset_path = context.get("dataset_path")
        if not dataset_path:
            raise ValueError("Dataset path is required")

        # Create the dataset
        dataset = LacrossePlayerDataset(
            image_dir=dataset_path, 
            transform=get_transforms('advanced_training'),
            min_images_per_player=1)
        dataset.crea

        return {"status": "success", "dataset_path": dataset_path}