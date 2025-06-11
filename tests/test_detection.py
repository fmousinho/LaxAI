# Test file for Detection Model in LaxAI.modules.detection
#
# Use pytest tests/test_detection.py to run the test file.
#
# Requires TEST_IMAGE_FILENAME to be present in the same directory as this test file.
# Requires the Store to be available (initialized automatically by the detection.py module).
# 
# Note several tests are marked as xfail because the RFDETRBase model does not yet support
# torch.Tensor, list of torch.Tensor, list of PIL.Image, or list of file paths as input. 

import pytest
import torch
import numpy as np
import supervision as sv
from PIL import Image
import os
import cv2 # Import cv2
import io
from unittest.mock import MagicMock, patch

# Assume the project root is the current working directory when running tests
# Adjust the import path if your project structure is different
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.detection import DetectionModel
from tools.store_driver import Store # Use full package path
from rfdetr import RFDETRBase # type: ignore

# --- Test Data ---
# Construct path to the test image relative to this test file's directory
_CURRENT_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_FILENAME = 'FCA_Upstate_NY_003_mp4-0001_jpg.rf.e23b522809c56498ddac9d7b7b82db2b.jpg'
TEST_IMAGE_PATH = os.path.join(_CURRENT_TEST_DIR, TEST_IMAGE_FILENAME)


# Define expected detections for the test image. Obtained by manipulating 
# _annotations.coco.json (the json output from COCO annotations in Roboflow)
EXPECTED_DETECTIONS_XYXY = np.array([
    [510, 219, 517, 240],
    [297, 213, 303, 239],
    [128, 219, 134, 248],
    [178, 232, 183, 262],
    [155, 213, 162, 234],
    [136, 223, 142, 249],
    [570, 221, 577, 246],
    [501, 231, 508, 259],
    [103, 229, 109, 256],
    [489, 215, 493, 236],
    [102, 217, 108, 238],
    [444, 229, 452, 262],
    [171, 228, 178, 259],
    [543, 220, 548, 245],
    [57, 256, 66, 301],
    [391, 208, 397, 229],
    [266, 224, 271, 252],
    [276, 223, 281, 255],
    [32, 219, 37, 243],
    [365, 207, 370, 224],
    [580, 222, 589, 246],
    [230, 208, 234, 226],
    [318, 224, 324, 252],
    [480, 245, 488, 287],
    [49, 233, 57, 265],
    [580, 233, 587, 265]
])
    
# Since confidence scores are not critical for this test's pass/fail criteria,
# we'll populate it with placeholder values to match the length of other arrays.
# The actual assertion for confidence will be removed.
EXPECTED_DETECTIONS_CONFIDENCE = np.zeros(len(EXPECTED_DETECTIONS_XYXY)) # Array of 26 zeros

EXPECTED_DETECTIONS_CLASS_ID = np.array([
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    3,
    5,
    5,
    3,
    3
]) # player_class_id=3, goalie is 1 and referee is 5)

# Create the expected sv.Detections object
EXPECTED_SVDETECTIONS = sv.Detections( # type: ignore
    xyxy=EXPECTED_DETECTIONS_XYXY,
    confidence=EXPECTED_DETECTIONS_CONFIDENCE,
    class_id=EXPECTED_DETECTIONS_CLASS_ID
)

# Load the test image in various formats for testing
@pytest.fixture(scope="module")
def test_image_formats():
    """Loads the test image in various formats."""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image not found at {TEST_IMAGE_PATH}")

    img_bgr = cv2.imread(TEST_IMAGE_PATH)
    if img_bgr is None:
         pytest.skip(f"Could not read test image from {TEST_IMAGE_PATH}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # NumPy array (RGB)
    img_np = img_rgb

    # PyTorch Tensor (CHW, float, normalized)
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1) / 255.0
    # For single tensor input, it might be expected as (C,H,W) or (1,C,H,W)
    # For list of tensors, each tensor in the list should be (C,H,W)

    # PIL Image (RGB)
    img_pil = Image.fromarray(img_rgb)

    # File Path (string)
    img_path = TEST_IMAGE_PATH

    # Create lists of each type (e.g., a batch of 2 for demonstration)
    list_size = 2
    list_img_np = [img_np.copy() for _ in range(list_size)]
    list_img_tensor = [img_tensor.clone() for _ in range(list_size)] # Each tensor is (C,H,W)
    list_img_pil = [img_pil.copy() for _ in range(list_size)]
    list_img_path = [img_path for _ in range(list_size)]

    return {
        "numpy": img_np,
        "torch_tensor": img_tensor,
        "pil_image": img_pil,
        "path": img_path,
        "list_numpy": list_img_np,
        "list_torch_tensor": list_img_tensor,
        "list_pil_image": list_img_pil,
        "list_path": list_img_path,
    }

# --- Fixtures for Mocks ---

@pytest.fixture
def real_store():
    """Provides a real Store object. Requires Google Drive authentication setup."""
    # This will attempt real authentication. Ensure credentials.json is available.
    # And that the test environment can perform the OAuth flow if needed.
    store = Store()
    if not store.is_initialized():
        pytest.skip("Real Store object could not be initialized. Check Google Drive credentials/authentication.")
    return store

@pytest.fixture
def mock_store_uninitialized():
    """Provides a mock Store object that is not initialized."""
    mock_store = MagicMock(spec=Store)
    mock_store.is_initialized.return_value = False
    return mock_store

# --- Test Cases ---

def test_detection_model_init_success(real_store):
    """
    Test successful initialization of DetectionModel using a real Store
    to download the checkpoint and initialize a real RFDETRBase model.
    """
    try:
        detection_model = DetectionModel(store=real_store, device=torch.device("cpu"))
        # Assert that the model attribute is an instance of RFDETRBase
        assert isinstance(detection_model.model, RFDETRBase)
        # You could add more assertions here, e.g., check if model parameters are on CPU
        # for param in detection_model.model.parameters():
        #     assert param.device.type == 'cpu'
        #     break # Check one parameter is enough
    except RuntimeError as e:
        if "Failed to load" in str(e) or "Store object is not initialized" in str(e):
            pytest.fail(f"DetectionModel initialization failed with real store: {e}")
        raise

def test_detection_model_init_store_uninitialized(mock_store_uninitialized):
    """Test initialization failure when Store is not initialized."""
    with pytest.raises(RuntimeError, match="Store object is not initialized"):
        DetectionModel(store=mock_store_uninitialized, device=torch.device("cpu"))

@pytest.mark.parametrize("image_format_param", [
    pytest.param("numpy", id="numpy_array_supported"),
    pytest.param("pil_image", id="pil_image_supported"),
    pytest.param("path", id="file_path_supported"),
    pytest.param("torch_tensor", id="torch_tensor_unsupported",
                 marks=pytest.mark.xfail(reason="torch.Tensor input not yet implemented in RF-DETR predict",
                                         raises=NotImplementedError, strict=True)),
    pytest.param("list_numpy", id="list_numpy_unsupported",
                 marks=pytest.mark.xfail(reason="List input not yet implemented in RF-DETR predict",
                                         raises=NotImplementedError, strict=True)),
    pytest.param("list_torch_tensor", id="list_torch_tensor_unsupported",
                 marks=pytest.mark.xfail(reason="List of torch.Tensor input not yet implemented in RF-DETR predict",
                                         raises=NotImplementedError, strict=True)),
    pytest.param("list_pil_image", id="list_pil_image_unsupported",
                 marks=pytest.mark.xfail(reason="List of PIL.Image input not yet implemented in RF-DETR predict",
                                         raises=NotImplementedError, strict=True)),
    pytest.param("list_path", id="list_path_unsupported",
                 marks=pytest.mark.xfail(reason="List of file paths input not yet implemented in RF-DETR predict",
                                         raises=NotImplementedError, strict=True)),
])
def test_generate_detections_all_input_types(real_store, test_image_formats, image_format_param):
    """
    Test generate_detections with various input image formats.
    Uses a real Store and real RFDETRBase model instance, but mocks the predict method.
    for supported types. Unsupported types are marked as xfail.
    """
    detection_model = DetectionModel(store=real_store, device=torch.device("cpu"))
    assert isinstance(detection_model.model, RFDETRBase) # Ensure real model was initialized

    # Get the image data in the specified format
    image_input = test_image_formats[image_format_param]
    
    # Mock the 'predict' method of the actual model instance created by DetectionModel
    # This mock is primarily for the "supported" cases. For "unsupported" (xfail) cases,
    # the NotImplementedError should be raised by generate_detections before predict is called.
    # However, having the mock active doesn't hurt and ensures that if an xfail case
    # unexpectedly passes the NotImplementedError check, it would then use a mocked predict.
    with patch.object(detection_model.model, 'predict', return_value=EXPECTED_SVDETECTIONS) as mock_predict_method:
        # This call will raise NotImplementedError for the xfail cases,
        # which pytest will handle based on the xfail marker.
        # For supported cases, it will proceed and use the mocked predict.
        detections = detection_model.generate_detections(image_input, threshold=0.6)

        # Assert that the mocked model.predict was called with the correct arguments
        # The 'images' argument passed to model.predict will be the preprocessed version
        # depending on how RFDETRBase.predict handles the input types.
        mock_predict_method.assert_called_once_with(
            image_input, 
            threshold=0.6
        )

        # These assertions will only be effectively checked for the non-xfail cases.
        # If an xfail case reaches here (i.e., it didn't raise NotImplementedError),
        # and these assertions pass, it will be an XPASS (and fail the suite due to strict=True).
        # If these assertions fail for an xfail case that didn't raise NotImplementedError,
        # it will still be an XFAIL (because it failed, as expected, just not with NotImplementedError).
        assert isinstance(detections, sv.Detections) # type: ignore
        np.testing.assert_array_equal(detections.xyxy, EXPECTED_SVDETECTIONS.xyxy)
        np.testing.assert_array_equal(detections.class_id, EXPECTED_SVDETECTIONS.class_id)

# You could add more tests here, e.g.:
# - test_generate_detections_batch_images (requires mocking predict to return List[sv.Detections])
# - test_generate_detections_no_detections (requires mocking predict to return empty sv.Detections)
# - test_load_model_file_not_found (requires mocking store.get_file_by_name to return None)
# - test_load_model_torch_load_error (requires mocking torch.load to raise an error)

# Helper to run tests (optional, you can just use pytest command)
if __name__ == "__main__":
    pytest.main([__file__])