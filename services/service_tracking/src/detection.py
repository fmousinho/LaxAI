import logging
logger = logging.getLogger(__name__)

import os
import tempfile
# Suppress WandB Scope.user deprecation warning
import warnings
from typing import Optional

import torch
from rfdetr import RFDETRBase  # type: ignore
import wandb

warnings.filterwarnings(
    "ignore", message=r".*The `Scope\.user` setter is deprecated.*", category=DeprecationWarning
)

WANDB_MODEL_ARTIFACT = "fmousinho76-home-org/wandb-registry-model/Detections:latest"
ARTIFCT_FILE_NAME = "common-models-detection_latest.pth"


class DetectionModel(RFDETRBase):
    """
    Manages the object detection model lifecycle, including loading from a
    specified store and performing inference.

    This class is responsible for abstracting the details of model interaction,
    providing a straightforward interface (`generate_detections`) to get
    predictions from images or image-like data.
    """

    def __init__(
        self,
        device: Optional[str] = None,
    ):
        """
        Initializes the DetectionModel.

        Args:
            device: The torch.device (cpu or cuda) to load the model onto.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        
        # Initializes wandb and downloads the model
        api_key = os.environ.get("WANDB_API_KEY")
        wandb.login(key=api_key)
        run_params = {
            "project": "LaxAI-Tracking",
            "name": "model-download",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with wandb.init(**run_params) as run:
                artifact = run.use_artifact(WANDB_MODEL_ARTIFACT, type="model")
                artifact_dir = artifact.download(root=tmpdir)
            model_path = os.path.join(artifact_dir, ARTIFCT_FILE_NAME)

            # Model was trained with 6 classes, and only later reduced to 3
            super().__init__(
                device=self.device,
                pretrain_weights=model_path,
                num_classes=6,
            )        



