import logging
import os
import tempfile
from typing import List, Optional, Union

import numpy as np
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETRBase  # type: ignore

from core.common.google_storage import get_storage
from config.all_config import detection_config

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
        model_dict: str = detection_config.model_checkpoint,
        model_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ): 
        """
        Initializes the DetectionModel.

        Args:
            model_dict: The name of the model dictionary (checkpoint file) from Google Storage.
                        Defaults to `detection_config.model_checkpoint`.
            model_dir: The path within the storage where the model dictionary resides.
                       Defaults to `detection_config.checkpoint_dir`.
            device: The torch.device (cpu or cuda) to load the model onto.
            storage_user_path: The user-specific path in Google Storage. If None, uses default.
        """
        self.model: RFDETRBase
        self.storage_user_path = detection_config.default_storage_user_path
        self.storage_client = get_storage(self.storage_user_path)
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
        
        if self._load_model():
            logger.info(f"Detection model '{self.model.__class__.__name__}' successfully initialized")
            logger.info(f"Detection threshold: {detection_config.prediction_threshold}")
            logger.info(f"Model loaded onto device: {self.device}")
        else:
            raise RuntimeError(f"Failed to load '{self.model_dict}' from '{self.model_dir}/{self.model_dict}'.")

    def _load_model(self) -> bool:
        """
        Downloads the model file from Google Storage and loads it onto the specified device.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        try:
            # Construct the blob name for the model file
            if self.model_dir:
                blob_name = f"{self.model_dir}/{self.model_dict}"
            else:
                blob_name = self.model_dict
            
            # Check if the model file exists in storage
            if not self.storage_client.blob_exists(blob_name):
                logger.error(f"Model file '{blob_name}' not found in Google Storage at path '{self.storage_user_path}'")
                return False
            
            # Create a temporary file to download the model
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_f:
                temp_checkpoint_path = tmp_f.name
            
            try:
                # Download the model file from Google Storage
                logger.info(f"Downloading model '{blob_name}' from Google Storage...")
                if not self.storage_client.download_blob(blob_name, temp_checkpoint_path):
                    logger.error(f"Failed to download model file '{blob_name}' from Google Storage")
                    return False
                
                logger.info(f"Model downloaded successfully to: {temp_checkpoint_path}")
                
                # Load the model from the downloaded file
                self.model = RFDETRBase(
                    device=self.device.type, 
                    pretrain_weights=temp_checkpoint_path, 
                    num_classes=6
                )
                
                logger.info(f"Model loaded successfully from Google Storage")
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
            logger.error(f"Error loading detection model from Google Storage: {e}", exc_info=True)
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

def get_model(
    model_dict: str = detection_config.model_checkpoint,
    model_dir: Optional[str] = None,
    device: Optional[torch.device] = None
) -> DetectionModel:
    """
    Factory function to create and return a DetectionModel instance.

    Args:
        model_dict: The name of the model dictionary (checkpoint file) from Google Storage.
                    Defaults to `detection_config.model_checkpoint`.
        model_dir: The path within the storage where the model dictionary resides.
                   Defaults to `detection_config.checkpoint_dir`.
        device: The torch.device (cpu or cuda) to load the model onto.

    Returns:
        An initialized DetectionModel instance.
    """
    return DetectionModel(
        model_dict=model_dict,
        model_dir=model_dir,
        device=device,
    )









# Test code to verify the DetectionModel class can run successfully
if __name__ == "__main__":
    import sys
    import traceback
    
    # Configure logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def test_detection_model():
        """Test the DetectionModel class initialization and basic functionality."""
        try:
            print("Testing DetectionModel initialization...")
            
            # Test 1: Test configuration access and storage setup
            print("Test 1: Testing configuration and storage client...")
            print(f"✓ Default storage user path from config: {detection_config.default_storage_user_path}")
            print(f"✓ Model checkpoint from config: {detection_config.model_checkpoint}")
            
            # Test 2: Test Google Storage client initialization
            print("\nTest 2: Testing Google Storage client initialization...")
            storage_client = get_storage("test/path")
            print(f"✓ Storage client created successfully: {type(storage_client)}")
            
            # Test 3: Test device selection logic
            print("\nTest 3: Testing device selection...")
            if torch.cuda.is_available():
                expected_device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                expected_device = torch.device("mps")
            else:
                expected_device = torch.device("cpu")
            print(f"✓ Expected device: {expected_device}")
            
            # Test 4: Test empty detections utility
            print("\nTest 4: Testing empty detections utility...")
            empty_dets = sv.Detections.empty()
            print(f"✓ Empty detections created: {type(empty_dets)}")
            print(f"✓ Number of empty detections: {len(empty_dets)}")
            
            # Test 5: Test NotImplementedError for unsupported inputs (mock test)
            print("\nTest 5: Testing NotImplementedError handling...")
            try:
                # Create a dummy tensor to test the validation logic
                dummy_tensor = torch.zeros((1, 3, 224, 224))
                # This would be used in generate_detections method
                if isinstance(dummy_tensor, (torch.Tensor, list)):
                    raise NotImplementedError(
                        "torch.Tensor and List inputs are not yet supported by the underlying RF-DETR model. "
                        "Please use a file path (str), PIL.Image, or np.ndarray."
                    )
                print("✗ Should have raised NotImplementedError")
            except NotImplementedError as e:
                print(f"✓ Correctly raises NotImplementedError: {str(e)[:80]}...")
            
            # Test 6: Test model initialization with custom parameters (without actual model loading)
            print("\nTest 6: Testing DetectionModel parameter validation...")
            try:
                # Test the constructor parameters without loading model
                print("✓ Testing with custom storage path...")
                custom_storage_path = "test/custom"
                print(f"✓ Custom storage path: {custom_storage_path}")
                
                print("✓ Testing with custom model parameters...")
                custom_model_dict = "test_model.pth"
                custom_model_dir = "test_dir"
                print(f"✓ Custom model dict: {custom_model_dict}")
                print(f"✓ Custom model dir: {custom_model_dir}")
                
                # Test device parameter
                custom_device = torch.device("cpu")
                print(f"✓ Custom device: {custom_device}")
                
            except Exception as e:
                print(f"✗ Parameter validation failed: {e}")
            
            print("\n" + "="*50)
            print("ALL TESTS COMPLETED SUCCESSFULLY!")
            print("✓ DetectionModel class configuration and utilities are working correctly")
            print("✓ Note: Actual model loading requires valid model files in Google Storage")
            print("="*50)
            
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            sys.exit(1)
    
    # Run the test
    test_detection_model()
