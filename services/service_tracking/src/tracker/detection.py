"""Detection model module for object detection using RFDETR."""

import logging
import os
import tempfile
import warnings
from typing import Optional

import torch
import wandb
from rfdetr import RFDETRBase  # type: ignore

from tracker.config import detection_model_config

logger = logging.getLogger(__name__)

# Suppress WandB Scope.user deprecation warning
warnings.filterwarnings(
    "ignore",
    message=r".*The `Scope\.user` setter is deprecated.*",
    category=DeprecationWarning
)


class DetectionModel(RFDETRBase):
    """Manages the object detection model lifecycle.

    This class handles loading and performing inference with the detection model,
    providing a straightforward interface to get predictions from images.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize the DetectionModel.

        Args:
            device: The torch device (cuda/mps/cpu) to load the model onto.
                   If None, auto-detected based on availability.
        """
        # Use provided device or get from config (which auto-detects if None)
        if device is None:
            if detection_model_config.device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = detection_model_config.device
        else:
            self.device = device

        # Initialize wandb and download the model using config values
        wandb_api_key = detection_model_config.wandb_api_key
        if wandb_api_key is None or wandb_api_key == "":
            raise ValueError(
                "WANDB_API_KEY is not set in DetectionModelConfig or "
                "environment variables."
            )

        wandb.login(key=detection_model_config.wandb_api_key)
        run_params = {
            "project": detection_model_config.wandb_project,
            "name": detection_model_config.wandb_run_name,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with wandb.init(**run_params) as run:
                artifact = run.use_artifact(
                    detection_model_config.wandb_model_artifact,
                    type="model"
                )
                artifact_dir = artifact.download(root=tmpdir)

            model_path = os.path.join(
                artifact_dir,
                detection_model_config.artifact_file_name
            )

            # Model was trained with 6 classes, and only later reduced to 3
            super().__init__(
                device=self.device,
                pretrain_weights=model_path,
                num_classes=detection_model_config.num_classes,
            )        



