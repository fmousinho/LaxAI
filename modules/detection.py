
import logging
from typing import Optional, Union, List
import os
import tempfile
import numpy as np
import torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase  # type: ignore
from tools.store_driver import Store

logger = logging.getLogger(__name__)

# Checkpoint for RFDETR Base model
MODEL_CHECKPOINT = "checkpoint.pth"
DEFAULT_CHECKPOINT_DIR = "Colab_Notebooks"
DEFAULT_TORCH_DEVICE = torch.device("cpu")
PREDICTION_THRESHOLD = 0.7


class DetectionModel:
    """
    Manages the object detection model lifecycle, including loading from a
    specified store and performing inference.

    This class is responsible for abstracting the details of model interaction,
    providing a straightforward interface (`generate_detections`) to get
    predictions from images or image-like data.
    """

    def __init__(self, 
                 store: Store,
                 model_dict: str = MODEL_CHECKPOINT,
                 model_dir: Optional[str] = DEFAULT_CHECKPOINT_DIR,
                 device: torch.device = DEFAULT_TORCH_DEVICE,
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
        temp_checkpoint_path = None
        try:
            checkpoint_buffer = self.store.get_file_by_name(file_name=self.model_dict, folder_path=self.model_dir)
            if checkpoint_buffer is None:
                logger.error(f"Failed to retrieve checkpoint buffer for '{self.model_dict}' from store.")
                return False

            # Create a named temporary file to save the checkpoint buffer
            # delete=False is important here so RFDETRBase can open it by name before we delete it.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_f: # Use appropriate suffix
                tmp_f.write(checkpoint_buffer.getvalue())
                temp_checkpoint_path = tmp_f.name # Get the path to the temporary file
            
            logger.info(f"Checkpoint saved to temporary file: {temp_checkpoint_path}")
            self.model = RFDETRBase(device=self.device.type, pretrain_weights=temp_checkpoint_path, num_classes=6)
            #self.model = self.model.optimize_for_inference()  # TODO: Must test and maybe uncomment this line
            return True
        except RuntimeError as e:
            logger.error(f"Error loading detection model: {e}", exc_info=True)
            return False
        finally:
            # Clean up the temporary file after the model has been loaded (or if an error occurred)
            if temp_checkpoint_path and os.path.exists(temp_checkpoint_path):
                try:
                    os.remove(temp_checkpoint_path)
                    logger.info(f"Temporary checkpoint file {temp_checkpoint_path} removed.")
                except OSError as e:
                    logger.error(f"Error removing temporary checkpoint file {temp_checkpoint_path}: {e}")

    def generate_detections(
            self,
            images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
            threshold: float = PREDICTION_THRESHOLD,
            **kwargs,
        ) -> sv.Detections: # type: ignore
        """
        Runs inference using the loaded detection model on the provided image(s).

        Args:
            images: The input image(s) to process. Can be a single image or a list/batch.
                    Accepts various formats: file path (str), PIL Image, NumPy array,
                    or PyTorch Tensor.
            threshold: Confidence threshold for detections.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      model's predict method.

        Returns:
            Results returned as `supervision.Detections` object or list or objects.
            The  format is determined by the underlying `RFDETRBase.predict` method.
            For more information on output format, refer to documentation:
            https://supervision.roboflow.com/latest/detection/core/
        """

        if isinstance(images, (torch.Tensor, list)):
            raise NotImplementedError(
                "torch.Tensor and List inputs are not yet supported by the underlying RF-DETR model.. "
                "Please use a file path (str), PIL.Image, or np.ndarray."
            )
        
        model_output_detections = self.model.predict(images, threshold=threshold, **kwargs)

        return model_output_detections
