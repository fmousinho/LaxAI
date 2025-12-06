import unittest
import sys
import os
from dotenv import load_dotenv

# Add the service_tracking directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
service_tracking_dir = os.path.dirname(current_dir)
sys.path.insert(0, service_tracking_dir)

# Load environment variables (expecting .env in project root)
# service_tracking_dir is .../services/service_tracking
# project root is .../LaxAI
project_root = os.path.dirname(os.path.dirname(service_tracking_dir))
load_dotenv(os.path.join(project_root, ".env"))

from src.detection import DetectionModel

class TestDetectionModelIntegration(unittest.TestCase):
    def test_model_loading_integration(self):
        """
        Integration test for Validation:
        1. Loads environment variables (API Key).
        2. Initializes DetectionModel (downloads from WandB).
        3. calls get_model() to verify model is loaded.
        """
        
        # Ensure API Key is present
        if not os.environ.get("WANDB_API_KEY"):
            self.skipTest("WANDB_API_KEY not found in environment. Skipping integration test.")
        
        # Initialize the model (This will trigger WandB download)
        # We use 'cpu' to avoid CUDA requirements on test runners if possible, 
        # unless the machine has GPU.
        model = DetectionModel(device="cpu")
        
        # Verify model is loaded using the requested method
        # User requested to use get_model_config instead of get_model
        model_config = model.get_model_config()
        
        self.assertIsNotNone(model_config, "Model config should be loaded and not None")
        # specific checks on model_config can be added here if needed
        # specific checks on loaded_model can be added here if we know the type (RFDETR)

if __name__ == '__main__':
    unittest.main()
