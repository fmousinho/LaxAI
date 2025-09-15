import logging
import os
import sys
import tempfile
# Suppress WandB Scope.user deprecation warning
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRBase  # type: ignore
from supervision import Detections

from shared_libs.config.all_config import detection_config, wandb_config
from shared_libs.utils.env_secrets import setup_environment_secrets

setup_environment_secrets()

import wandb

warnings.filterwarnings(
    "ignore", message=r".*The `Scope\.user` setter is deprecated.*", category=DeprecationWarning
)

logger = logging.getLogger(__name__)

REGISTRY_NAME = "wandb-registry-model"
COLLECTION_NAME = "Detections"
ALIAS = "latest"


class DetectionModel:
    """
    Manages the object detection model lifecycle, including loading from a
    specified store and performing inference.

    This class is responsible for abstracting the details of model interaction,
    providing a straightforward interface (`generate_detections`) to get
    predictions from images or image-like data.
    """

    def __init__(
        self,
        registry: str = REGISTRY_NAME,
        model_dir: str = COLLECTION_NAME,
        model_version: str = ALIAS,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the DetectionModel.

        Args:
            registry: The name of the model registry.
            device: The device to run the model on (CPU or GPU).
            device: The torch.device (cpu or cuda) to load the model onto.
        """
        self.model: RFDETRBase
        self.model_artifact = f"{registry}/{model_dir}:{model_version}"
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        if self._load_model():
            logger.info(
                f"Detection model '{self.model.__class__.__name__}' " "successfully initialized"
            )
            logger.info(f"Detection threshold: {detection_config.prediction_threshold}")
            logger.info(f"Model loaded onto device: {self.device}")
        else:
            raise RuntimeError(f"Failed to load '{self.model_artifact}' from wandb.")

    def _load_model(self) -> bool:
        """
        Downloads the model file from the GCS store and loads it onto the specified device.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        temp_checkpoint_path = None

        wandb_api_key = os.getenv("WANDB_API_KEY")

        if not wandb_api_key:
            logger.error("WandB API key not found in any source.")
            return False
        else:
            logger.info("WandB API key successfully loaded.")
        wandb.login(key=wandb_api_key)

        try:
            run = wandb.init(
                entity=wandb_config.team, project=wandb_config.project, job_type="download-model"
            )

            logger.info(f"Attempting to fetch artifact: {self.model_artifact}")
            fetched_artifact = run.use_artifact(artifact_or_name=self.model_artifact, type="model")

            # Create a temporary directory to download the model to.
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Downloading artifact to directory: {temp_dir}")
                download_path = fetched_artifact.download(root=temp_dir)

                if not download_path:
                    logger.error(f"Failed to download artifact: {self.model_artifact}")
                    return False

                # Log the contents of the download directory
                downloaded_files = os.listdir(temp_dir)
                logger.info(f"Files in download directory: {downloaded_files}")

                # Dynamically locate the model file
                for file_name in downloaded_files:
                    if file_name.endswith(".pth"):
                        temp_checkpoint_path = os.path.join(temp_dir, file_name)
                        break
                else:
                    logger.error("No .pth file found in the downloaded artifact.")
                    return False

                logger.info(f"Checkpoint file located at: {temp_checkpoint_path}")

                self.model = RFDETRBase(
                    device=self.device.type,
                    pretrain_weights=temp_checkpoint_path,
                    num_classes=6,
                )

                self.model.optimize_for_inference()

                return True

        except ConnectionError as comm_err:  # Communication error (originally wandb.errors.CommError)
            logger.error(f"Communication error with wandb: {comm_err}", exc_info=True)
            return False
        except ValueError as usage_err:  # Usage/configuration error (originally wandb.errors.UsageError)
            logger.error(f"Usage error with wandb: {usage_err}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error while loading detection model: {e}", exc_info=True)
            return False

        finally:
            # Ensure the temporary file is cleaned up
            if temp_checkpoint_path and os.path.exists(temp_checkpoint_path):
                try:
                    os.remove(temp_checkpoint_path)
                    logger.debug(f"Temporary checkpoint file removed: {temp_checkpoint_path}")
                except OSError as e:
                    logger.warning(
                        f"Error removing temporary checkpoint file " f"{temp_checkpoint_path}: {e}"
                    )

    def generate_detections(
        self,
        images: Union[
            str,
            Image.Image,
            np.ndarray,
            torch.Tensor,
            List[Union[str, np.ndarray, Image.Image, torch.Tensor]],
        ],
        threshold: float = detection_config.prediction_threshold,
        **kwargs,
    ) -> Union[Detections, List[Detections]]:
        """
        Runs inference using the loaded detection model on the provided image(s).

        Args:
            images: The input image(s) to process. Can be a single image or a list/batch.
                   Accepts various formats: file path (str), PIL Image, NumPy array,
                   or PyTorch Tensor.
            threshold: Confidence threshold for detections.
            **kwargs: Additional keyword arguments passed to the underlying model's predict method.

        Returns:
            Results as `supervision.Detections` object.

        Raises:
            NotImplementedError: If images is a torch.Tensor or list (not yet supported).
        """
        if isinstance(images, (torch.Tensor, list)) and sys.platform == "darwin":
            raise NotImplementedError(
                "torch.Tensor and List inputs are not yet supported by the underlying "
                "RF-DETR model. Please use a file path (str), PIL.Image, or np.ndarray."
            )

        return self.model.predict(images, threshold=threshold, **kwargs)

    def empty_detections(self):
        """Returns an empty Detections object."""
        return Detections.empty()
