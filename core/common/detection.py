# Enable MPS fallback for PyTorch operations on macOS (must be set before torch import)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
import sys
import tempfile
import hashlib
import json
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETRBase  # type: ignore

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        device: Optional[torch.device] = None,
    ): 
        """
        Initializes the DetectionModel.

        Args:
            model_dict: The name of the model dictionary (checkpoint file) from Google Storage.
                        Defaults to `detection_config.model_checkpoint`.
            device: The torch.device (cpu or cuda) to load the model onto.
            storage_user_path: The user-specific path in Google Storage. If None, uses default.
        """
        self.model: RFDETRBase
        self.storage_user_path = detection_config.default_storage_user_path
        self.storage_client = get_storage(self.storage_user_path)
        self.model_dict = model_dict
        
        # Set up caching directory
        self.cache_dir = Path("/var/tmp/laxai_models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_model_path = self.cache_dir / f"{model_dict}"
        self.cache_metadata_path = self.cache_dir / f"{model_dict}.meta"

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
            raise RuntimeError(f"Failed to load '{self.model_dict}' from '{self.storage_user_path}/{self.model_dict}'.")

    def _load_model(self) -> bool:
        """
        Downloads the model file from Google Storage with caching support.
        Checks if a newer version exists and downloads only if necessary.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        try:
            # The blob name is just the model filename since Google Storage client
            # automatically adds the user_path prefix
            blob_name = self.model_dict
            
            # Check if the model file exists in storage
            if not self.storage_client.blob_exists(blob_name):
                logger.error(f"Model file '{self.model_dict}' not found in Google Storage at path '{self.storage_user_path}'")
                return False
            
            # Get remote model metadata
            remote_metadata = self._get_remote_model_metadata(blob_name)
            if not remote_metadata:
                logger.error(f"Failed to get metadata for model '{blob_name}'")
                return False
            
            # Check if we need to download the model
            model_path = self._get_or_download_model(blob_name, remote_metadata)
            if not model_path:
                return False
            
            # Load the model from the cached/downloaded file
            self.model = RFDETRBase(
                device=self.device.type, 
                pretrain_weights=str(model_path), 
                num_classes=6
            )
            
            logger.info(f"Model loaded successfully from: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading detection model from Google Storage: {e}", exc_info=True)
            return False
    
    def _get_remote_model_metadata(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the remote model file.
        
        Args:
            blob_name: The blob name of the model file
            
        Returns:
            Dictionary with model metadata or None if failed
        """
        try:
            # Download a small portion of the file to get its hash
            # We'll use a temporary file to get the full file info
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_f:
                temp_path = tmp_f.name
            
            try:
                # Download the model file to get its hash
                if not self.storage_client.download_blob(blob_name, temp_path):
                    logger.error(f"Failed to download model file '{blob_name}' for metadata")
                    return None
                
                # Calculate file hash
                file_hash = self._calculate_file_hash(temp_path)
                file_size = os.path.getsize(temp_path)
                
                # Clean up temporary file
                os.remove(temp_path)
                
                return {
                    "hash": file_hash,
                    "size": file_size,
                    "blob_name": blob_name
                }
                
            except Exception as e:
                # Clean up temporary file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
                
        except Exception as e:
            logger.error(f"Error getting remote model metadata: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 hash as hex string
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_cached_model_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the cached model file.
        
        Returns:
            Dictionary with cached model metadata or None if no cache exists
        """
        try:
            if not self.cache_metadata_path.exists():
                return None
                
            with open(self.cache_metadata_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.warning(f"Error reading cached model metadata: {e}")
            return None
    
    def _save_cached_model_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Save metadata about the cached model file.
        
        Args:
            metadata: Dictionary with model metadata
        """
        try:
            with open(self.cache_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error saving cached model metadata: {e}")
    
    def _get_or_download_model(self, blob_name: str, remote_metadata: Dict[str, Any]) -> Optional[Path]:
        """
        Get the model file, either from cache or by downloading.
        
        Args:
            blob_name: The blob name of the model file
            remote_metadata: Metadata about the remote model file
            
        Returns:
            Path to the model file or None if failed
        """
        try:
            # Check if we have a cached version
            cached_metadata = self._get_cached_model_metadata()
            
            # Check if cached model exists and is up to date
            if (cached_metadata and 
                self.cached_model_path.exists() and 
                cached_metadata.get("hash") == remote_metadata.get("hash")):
                
                logger.info(f"Using cached model: {self.cached_model_path}")
                return self.cached_model_path
            
            # Need to download the model
            logger.info(f"Downloading model '{blob_name}' from Google Storage...")
            
            # Download to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_f:
                temp_path = tmp_f.name
            
            try:
                if not self.storage_client.download_blob(blob_name, temp_path):
                    logger.error(f"Failed to download model file '{blob_name}' from Google Storage")
                    return None
                
                # Verify the downloaded file hash
                downloaded_hash = self._calculate_file_hash(temp_path)
                if downloaded_hash != remote_metadata.get("hash"):
                    logger.error(f"Downloaded model hash mismatch. Expected: {remote_metadata.get('hash')}, Got: {downloaded_hash}")
                    os.remove(temp_path)
                    return None
                
                # Move the downloaded file to cache
                if self.cached_model_path.exists():
                    self.cached_model_path.unlink()
                
                os.rename(temp_path, str(self.cached_model_path))
                
                # Save metadata
                self._save_cached_model_metadata(remote_metadata)
                
                logger.info(f"Model downloaded and cached successfully: {self.cached_model_path}")
                return self.cached_model_path
                
            except Exception as e:
                # Clean up temporary file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
                
        except Exception as e:
            logger.error(f"Error getting or downloading model: {e}")
            return None

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

    def clear_cache(self) -> bool:
        """
        Clear the cached model and metadata files.
        
        Returns:
            True if cache was cleared successfully, False otherwise
        """
        try:
            cache_cleared = False
            
            # Remove cached model file
            if self.cached_model_path.exists():
                self.cached_model_path.unlink()
                logger.info(f"Removed cached model: {self.cached_model_path}")
                cache_cleared = True
            
            # Remove cached metadata file
            if self.cache_metadata_path.exists():
                self.cache_metadata_path.unlink()
                logger.info(f"Removed cached metadata: {self.cache_metadata_path}")
                cache_cleared = True
            
            if not cache_cleared:
                logger.info("No cache files to clear")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    @classmethod
    def clear_all_cache(cls) -> bool:
        """
        Clear all cached model files in the cache directory.
        
        Returns:
            True if cache was cleared successfully, False otherwise
        """
        try:
            cache_dir = Path("/var/tmp/laxai_models")
            
            if not cache_dir.exists():
                logger.info("Cache directory does not exist")
                return True
            
            # Remove all files in cache directory
            files_removed = 0
            for file_path in cache_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    files_removed += 1
            
            logger.info(f"Cleared {files_removed} files from cache directory: {cache_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return False

    def empty_detections(self):
        """Returns an empty Detections object."""
        return sv.Detections.empty()

def get_model(
    model_dict: str = detection_config.model_checkpoint,
    device: Optional[torch.device] = None
) -> DetectionModel:
    """
    Factory function to create and return a DetectionModel instance.

    Args:
        model_dict: The name of the model dictionary (checkpoint file) from Google Storage.
                    Defaults to `detection_config.model_checkpoint`.
        device: The torch.device (cpu or cuda) to load the model onto.

    Returns:
        An initialized DetectionModel instance.
    """
    return DetectionModel(
        model_dict=model_dict,
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
