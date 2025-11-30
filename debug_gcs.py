import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set env vars
os.environ["GOOGLE_CLOUD_PROJECT"] = "laxai-466119"
os.environ["LOG_LEVEL"] = "DEBUG"

# Add path to sys.path
sys.path.append(os.getcwd())

# Mock missing dependencies
from unittest.mock import MagicMock
sys.modules["cv2"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["yaml"] = MagicMock()

from shared_libs.common.google_storage import GoogleStorageClient

def test_gcs():
    tenant_id = "tenant1"
    bucket_name = "laxai_dev"
    
    print(f"Initializing GoogleStorageClient with tenant_id={tenant_id}")
    client = GoogleStorageClient(tenant_id=tenant_id)
    
    # Manually set bucket name if needed, but it should be loaded from config
    # We might need to mock config if it's not loading correctly
    # But let's try with default loading first
    
    dataset_address = "laxai_dev/tenant1/datasets/M&D Orlando 27 Black vs GRIT Dallas_summary"
    print(f"Testing dataset address: {dataset_address}")
    
    # Normalize logic (copied from TrainingController)
    normalized_address = dataset_address
    if bucket_name and normalized_address.startswith(f"{bucket_name}/"):
        normalized_address = normalized_address[len(bucket_name) + 1:]
    
    if normalized_address.startswith(f"{tenant_id}/"):
        normalized_address = normalized_address[len(tenant_id) + 1:]
        
    print(f"Normalized address: {normalized_address}")
    
    # Add /train/ suffix
    train_path = normalized_address.rstrip("/") + "/train/"
    print(f"Train path to query: {train_path}")
    
    # List blobs
    print(f"Listing blobs with prefix={train_path} and delimiter='/'")
    try:
        blobs = client.list_blobs(prefix=train_path, delimiter='/')
        print(f"Found {len(blobs)} items")
        for blob in blobs:
            print(f" - {blob}")
            
        # If blobs found, try listing one player
        if blobs:
            player = list(blobs)[0]
            print(f"Listing player: {player}")
            player_blobs = client.list_blobs(prefix=player)
            print(f"Found {len(player_blobs)} images for player")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gcs()
