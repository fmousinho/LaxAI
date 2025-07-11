
import logging
import os
import tempfile
from typing import List, Optional, Union

import numpy as np
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETRBase  # type: ignore

from config.transforms_config import detection_config
from tools.store_driver import Store

logger = logging.getLogger(__name__)


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
        store: Store,
        model_dict: str = detection_config.model_checkpoint,
        model_dir: Optional[str] = detection_config.checkpoint_dir,
        device: Optional[torch.device] = None,
    ): 
        """
        Initializes the DetectionModel.

        Args:
            store: An initialized Store object for file access. This is required for model loading.
            model_dict: The name of the model dictionary (checkpoint file) from the store object.
                        Defaults to `MODEL_CHECKPOINT`.
            model_dir: The path within the store object where the model dictionary resides.
                       Defaults to `DEFAULT_CHECKPOINT_DIR`. # type: ignore
            device: The torch.device (cpu or cuda) to load the model onto.
        """
        self.model: RFDETRBase
        self.store = store
        self.model_dict = model_dict
        self.model_dir = model_dir
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        if not store.is_initialized():
            raise RuntimeError("Store object is not initialized. Please initialize it before using DetectionModel.")
        
        if self._load_model():
            logger.info(f"Detection model '{self.model.__class__.__name__}' successfully initialized")
        else:
            raise RuntimeError(f"Failed to load '{self.model_dict}' from '{self.model_dir}/{self.model_dict}'.")

    def _load_model(self) -> bool:
        """
        Downloads the model file from the store provided and loads it onto the specified device.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        try:
            checkpoint_buffer = self.store.get_file_by_name(
                file_name=self.model_dict, folder_path=self.model_dir
            )
            if checkpoint_buffer is None:
                logger.error(f"Failed to retrieve checkpoint buffer for '{self.model_dict}' from store.")
                return False

            # Use context manager for proper cleanup
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_f:
                tmp_f.write(checkpoint_buffer.getvalue())
                temp_checkpoint_path = tmp_f.name

            try:
                logger.info(f"Checkpoint saved to temporary file: {temp_checkpoint_path}")
                self.model = RFDETRBase(
                    device=self.device.type, 
                    pretrain_weights=temp_checkpoint_path, 
                    num_classes=6
                )
                logger.info(f"Detection model '{self.model.__class__.__name__}' successfully loaded")
                return True
            finally:
                # Ensure cleanup even if model loading fails
                if os.path.exists(temp_checkpoint_path):
                    try:
                        os.remove(temp_checkpoint_path)
                        logger.debug(f"Temporary checkpoint file removed: {temp_checkpoint_path}")
                    except OSError as e:
                        logger.warning(f"Error removing temporary checkpoint file {temp_checkpoint_path}: {e}")

        except Exception as e:
            logger.error(f"Error loading detection model: {e}", exc_info=True)
            return False

    def generate_detections(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
        threshold: float = detection_config.prediction_threshold,
        **kwargs,
    ) -> sv.Detections:
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
        if isinstance(images, (torch.Tensor, list)):
            raise NotImplementedError(
                "torch.Tensor and List inputs are not yet supported by the underlying RF-DETR model. "
                "Please use a file path (str), PIL.Image, or np.ndarray."
            )

        return self.model.predict(images, threshold=threshold, **kwargs)

    def empty_detections(self):
        """Returns an empty Detections object."""
        return sv.Detections.empty()
