"""
Google Cloud Storage utilities for the LaxAI project.

This module provides a client and helpers for interacting with Google Cloud Storage,
including error handling, credential management, and common operations.
"""
#===========================================================
# Google Cloud Storage Client with Error Handling and Common Operations
#
# This module is hardcoded to a specific project and bucket for simplicity.
#
#
# TODO:
# - Make sure credentials in Google console are restricted enough
# - Add support for multi-tenancy
#===========================================================




import logging
import os
import yaml
import tempfile
import cv2
import json
import numpy as np
import io 
from typing import Optional, List, Any
import PIL.Image
from supervision import Detections, JSONSink
from utils.env_or_colab import load_env_or_colab
from google.cloud import storage
from google.cloud.storage import Blob
from google.cloud.exceptions import NotFound, Forbidden
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account
from config.all_config import google_storage_config

logger = logging.getLogger(__name__)

class GCSPaths:
    """
    A class to manage Google Cloud Storage paths defined in a YAML file.
    """
    def __init__(self):
        self.project_id = google_storage_config.project_id
        self.bucket_name = google_storage_config.bucket_name
        self.gcs_paths_file = google_storage_config.gcs_paths_file
        self.paths = self._load_paths()


    def _load_paths(self) -> dict:
        """Loads paths from the YAML configuration file."""
        try:
            # Handle absolute path construction
            if self.gcs_paths_file.startswith('/config/'):
                # Extract filename from the path
                config_filename = os.path.basename(self.gcs_paths_file)
                # Get the absolute path to this file and replace the last two components
                # This file is at: /full/path/to/LaxAI/src/common/google_storage.py
                # We want:      /full/path/to/LaxAI/src/config/gcs_structure.yaml
                current_file_path = os.path.abspath(__file__)
                # Replace 'common/google_storage.py' with 'config/gcs_structure.yaml'
                src_dir = os.path.dirname(os.path.dirname(current_file_path))  # Get src directory
                actual_path = os.path.join(src_dir, 'config', config_filename)
            else:
                actual_path = os.path.abspath(self.gcs_paths_file)
                
            logger.debug(f"Loading GCS paths from: {actual_path}")
            
            with open(actual_path, 'r') as f:
                config = yaml.safe_load(f)
                # Extract the data_prefixes from the nested structure
                return config.get('gcs', {}).get('data_prefixes', {})
        except FileNotFoundError:
            logger.error(f"GCS paths configuration file not found: {actual_path if 'actual_path' in locals() else self.gcs_paths_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing GCS paths YAML file: {e}")
            raise

    def get_path(self, key: str, **kwargs) -> str:
        """
        Retrieves a formatted path string using the given key and keyword arguments.

        Args:
            key: The key corresponding to the desired path in the YAML file.
            **kwargs: Keyword arguments to format the path string (e.g., tenant_id="tenant1", video_id="vid123").

        Returns:
            The formatted path string.

        Raises:
            KeyError: If the key is not found in the configuration.
        """
        path_template = self.paths.get(key)
        if path_template is None:
            raise KeyError(f"Path key '{key}' not found in {self.gcs_paths_file}")
        try:
            return path_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required argument for path '{key}': {e}")
            return None
        except Exception as e:
            logger.error(f"Error formatting path '{key}': {e}")
            raise




class GoogleStorageClient:
    """Google Cloud Storage client with error handling and common operations."""

    def __init__(self, tenant_id: str, credentials: Optional[service_account.Credentials]=None):
        """
        Initialize the Google Storage client with predefined configuration.
        
        Args:
            tenant_id: The unique identifier for the tenant (e.g., "tenant1")
            user_id: The unique identifier for the user (e.g., "tenant1/user123")
        """
        self.config = google_storage_config
        self.user_id = tenant_id
        self._client = None
        self._bucket = None
        self._authenticated = False
        self.credentials = credentials
        
    def _authenticate(self) -> bool:
        """
        Authenticate with Google Cloud Storage.
        
        Raises:
            RuntimeError: If authentication fails
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # If credentials object is provided, use it directly
            if self.credentials:
                self._client = storage.Client(credentials=self.credentials, project=self.config.project_id)
                logger.info(f"Using provided service account credentials for project: {self.config.project_id}")
            else:
                # Check if credentials file path is set in environment
                if self.config.credentials_name not in os.environ:
                    raise ValueError(f"{self.config.credentials_name} not set in environment variables")
                
                credentials_path = os.environ[self.config.credentials_name]
                
                # Create client with explicit project ID if provided
                if self.config.project_id:
                    self._client = storage.Client.from_service_account_json(
                        credentials_path, 
                        project=self.config.project_id
                    )
                    logger.info(f"Using service account file for project: {self.config.project_id}")
                else:
                    # Use default project from service account file
                    self._client = storage.Client.from_service_account_json(credentials_path)
                    logger.info(f"Using service account file with default project: {self._client.project}")

            # Test authentication by trying to get bucket
            self._bucket = self._client.bucket(self.config.bucket_name)
            
            # Test bucket access
            self._bucket.reload()
            
            self._authenticated = True
            logger.info(f"Successfully authenticated with Google Cloud Storage for bucket: {self.config.bucket_name}")
            return True
            
        except DefaultCredentialsError as e:
            logger.error(f"Authentication failed - No valid credentials found:")
            logger.error("Make sure to set GOOGLE_APPLICATION_CREDENTIALS in your .env file or environment variables.") 
            return False
        except NotFound as e:
            logger.error(f"Authentication failed - Bucket '{self.config.bucket_name}' not found: {e}")
            return False
        except Forbidden as e:
            logger.error(f"Authentication failed - Access denied to bucket '{self.config.bucket_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Authentication failed - Unexpected error: {e}")
            return False
    
    def _ensure_authenticated(self) -> bool:
        """Ensure client is authenticated before operations."""
        if not self._authenticated:
            return self._authenticate()
        return True

    def list_blobs(self, prefix: Optional[str] = None, include_user_id: bool = True) -> List[str]:
        """
        Lists all the blobs in the bucket.
        
        Args:
            prefix: Optional prefix to filter blobs (will be combined with user_path)
            include_user_id: Whether to include user_id prefix in the blob names

        Returns:
            List of blob names, inclusive of user_id prefix if set
            
        Raises:
            RuntimeError: If authentication fails
        """
        if not self._ensure_authenticated():
            raise RuntimeError("Failed to authenticate with Google Cloud Storage")
        
        try:
            # Add user_id prefix to search prefix, consistent with upload methods
            if self.user_id and not self.user_id.endswith('/'):
                self.user_id += '/'
            if prefix:
                if self.user_id:
                    full_prefix = f"{self.user_id}{prefix}"
                else:
                    full_prefix = prefix
            else:
                full_prefix = self.user_id if self.user_id else None
                
            blobs = self._client.list_blobs(self.config.bucket_name, prefix=full_prefix)
            user_id_len = len(self.user_id) if self.user_id else 0
            if include_user_id or user_id_len == 0:
                return [blob.name for blob in blobs]
            else:
                return [blob.name[user_id_len:] for blob in blobs]
          
        except Exception as e:
            logger.error(f"Failed to list blobs: {e}")
            raise

    def upload_blob(self, data: Any, destination_blob_name: str, data_type: str = 'file') -> bool:
        """
        Upload a file to the bucket.
        
        Args:
            source_file_path: Path to the source file
            destination_blob_name: Name of the blob in the bucket (will be prefixed with user_path)
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:

            if data_type == 'file':
                if type(data) is str:
                    return self.upload_from_file(destination_blob_name, data)
                else:
                    logger.error("Invalid data type for file upload. Expected string path.")
                    return False
            
            else:
                logger.error("Invalid data type for upload.")
                return False
            
        except Exception as e:
            logger.error(f"Failed to upload blob: {e}")
            return False
        
    def upload_from_bytes(self, destination_blob_name: str, data: bytes, content_type: Optional[str] = None) -> bool:
        """
        Upload bytes to a blob.
        
        Args:
            destination_blob_name: Name of the blob in the bucket (will be prefixed with user_path)
            data: Bytes data to upload
            content_type: Optional content type of the blob

        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            # Add user_path prefix to destination
            if len(self.user_id) > 0 and not self.user_id.endswith('/'):
                self.user_id += '/'
            if destination_blob_name.startswith(self.user_id):
                full_destination = destination_blob_name
            else:
                full_destination = f"{self.user_id}{destination_blob_name}"
            blob = self._bucket.blob(full_destination)

            if destination_blob_name.endswith('.jpg') or destination_blob_name.endswith('.jpeg'):
                bgr_data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                content_type = content_type or "image/jpeg"
                image_bytes = cv2.imencode(".jpg", bgr_data)[1].tobytes()
                blob.upload_from_string(image_bytes, content_type=content_type)
                return True

            elif isinstance(data, Detections):
                # Use JSONSink for proper serialization with a temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    temp_json_path = temp_file.name
                
                # Use JSONSink to serialize the Detection object
                json_sink = JSONSink(temp_json_path)
                with json_sink as sink:
                    sink.append(data)
                
                # Read the serialized JSON data
                with open(temp_json_path, 'r') as f:
                    json_string = f.read()
                
                # Clean up temporary file
                os.remove(temp_json_path)
                
                # Upload the JSON string
                blob.upload_from_string(json_string, content_type="application/json")
                return True


            elif destination_blob_name.endswith('.json'):
                json_bytes = json.dumps(data).encode("utf-8")
                content_type = content_type or "application/json"
                blob.upload_from_string(json_bytes, content_type=content_type)
                return True
            
            else:
                logger.error(f"Invalid file type for {destination_blob_name}.")
                return False
            
        except Exception as e:
            logger.error(f"Failed to upload blob from bytes: {e}")
            return False


    def download_blob(self, source_blob_name: str, destination_file_path: str) -> bool:
        """
        Download a blob from the bucket.
        
        Args:
            source_blob_name: Name of the blob in the bucket (will be prefixed with user_path)
            destination_file_path: Path to save the downloaded file
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            # Get the blob object and download it
            # If the source_blob_name already starts with user_path, don't prefix it again
            if self.user_id and source_blob_name.startswith(self.user_id + "/"):
                full_source = source_blob_name
            else:
                full_source = f"{self.user_id}/{source_blob_name}" if self.user_id else source_blob_name
            blob = self._bucket.blob(full_source)
            blob.download_to_filename(destination_file_path)
            logger.debug(f"Blob {full_source} downloaded to {destination_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download blob: {e}")
            return False
    
    def download_as_string(self, source_blob_name: str) -> Optional[str]:
        """
        Download a blob from the bucket as a string.
        
        Args:
            source_blob_name: Name of the blob in the bucket (will be prefixed with user_path)
            
        Returns:
            str: Content of the blob as string, or None if download failed
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return None
        
        try:
            # Add user_path prefix to source, but avoid double prefixing
            if self.user_id and source_blob_name.startswith(self.user_id + "/"):
                full_source = source_blob_name
            else:
                full_source = f"{self.user_id}/{source_blob_name}" if self.user_id else source_blob_name
            blob = self._bucket.blob(full_source)
            content = blob.download_as_text()
            logger.info(f"Blob {full_source} downloaded as string")
            return content
        except Exception as e:
            logger.error(f"Failed to download blob as string: {e}")
            return None
    
    def upload_from_string(self, destination_blob_name: str, data: str) -> bool:
        """
        Upload string data to a blob.
        
        Args:
            destination_blob_name: Name of the blob in the bucket (will be prefixed with user_path)
            data: String data to upload
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            # Add user_path prefix to destination
            if len(self.user_id) > 0 and not self.user_id.endswith('/'):
                self.user_id += '/'
            if destination_blob_name.startswith(self.user_id):
                full_destination = destination_blob_name
            else:     
                full_destination = f"{self.user_id}{destination_blob_name}"
            blob = self._bucket.blob(full_destination)
            blob.upload_from_string(data)
            logger.info(f"String data uploaded to {full_destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload string data: {e}")
            return False
    
    def upload_from_file(self, destination_blob_name: str, file_path: str) -> bool:
        """
        Upload a file to the bucket (alias for upload_blob for consistency).
        
        Args:
            destination_blob_name: Name of the blob in the bucket (will be prefixed with user_path)
            file_path: Path to the file to upload
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            # Add user_path prefix to destination
            if len(self.user_id) > 0 and not self.user_id.endswith('/'):
                self.user_id += '/'
            if destination_blob_name.startswith(self.user_id):
                full_destination = destination_blob_name
            else:
                full_destination = f"{self.user_id}{destination_blob_name}"
            blob = self._bucket.blob(full_destination)
            blob.upload_from_filename(file_path)
            logger.debug(f"File {file_path} uploaded to {full_destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False
    
    def copy_blob(self, source_blob_name: str, destination_blob_name: str) -> bool:
        """
        Copy a blob within the bucket.
        
        Args:
            source_blob_name: Name of the source blob (will be prefixed with user_path)
            destination_blob_name: Name of the destination blob (will be prefixed with user_path)
            
        Returns:
            bool: True if copy successful, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            if self.user_id and not self.user_id.endswith('/'):
                self.user_id += '/'
            if source_blob_name.startswith(self.user_id):
                full_source = source_blob_name
            else:
                full_source = f"{self.user_id}{source_blob_name}" if self.user_id else source_blob_name
            
            
            if destination_blob_name.startswith(self.user_id):
                full_destination = destination_blob_name
            else:
                full_destination = f"{self.user_id}{destination_blob_name}"

            source_blob = self._bucket.blob(full_source)
            new_blob = self._bucket.copy_blob(source_blob, self._bucket, new_name=full_destination)
            logger.debug(f"Blob {full_source} copied to {full_destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy blob: {e}")
            return False
    
    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from the bucket.
        
        Args:
            blob_name: Name of the blob to delete (will be prefixed with user_path)
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            # Add user_path prefix to blob name
            full_blob_name = f"{self.user_id}/{blob_name}"
            blob = self._bucket.blob(full_blob_name)
            blob.delete()
            logger.info(f"Blob {full_blob_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete blob: {e}")
            return False
    
    def blob_exists(self, blob_name: str) -> bool:
        """
        Check if a blob exists in the bucket.
        
        Args:
            blob_name: Name of the blob to check (will be prefixed with user_path)
            
        Returns:
            bool: True if blob exists, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            # Add user_path prefix to blob name
            full_blob_name = f"{self.user_id}/{blob_name}"
            blob = self._bucket.blob(full_blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check if blob exists: {e}")
            return False
        

    def move_blob(self, source_blob_name: str, destination_blob_name: str) -> bool:
        """
        Move a blob from one location to another within the bucket.
        
        Args:
            source_blob_name: Name of the source blob (will be prefixed with user_path)
            destination_blob_name: Name of the destination blob (will be prefixed with user_path)
            
        Returns:
            bool: True if move successful, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            # Add user_path prefix to both source and destination
            if self.user_id and not self.user_id.endswith('/'):
                self.user_id += '/'
            if source_blob_name.startswith(self.user_id):
                full_source = source_blob_name
            else:
                full_source = f"{self.user_id}{source_blob_name}"

            if destination_blob_name.startswith(self.user_id):
                full_destination = destination_blob_name
            else: 
                full_destination = f"{self.user_id}{destination_blob_name}"
            
            # Get source blob
            source_blob = self._bucket.blob(full_source)

            # Check if source blob exists
            if not source_blob.exists():
                logger.error(f"Source blob {full_source} does not exist")
                return False
            
            # Copy the blob to the new destination
            new_blob = self._bucket.copy_blob(source_blob, self._bucket, new_name=full_destination)
            
            # Delete the original blob
            source_blob.delete()
            
            logger.debug(f"Blob moved from {full_source} to {full_destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move blob: {e}")
            return False
        
    def get_video_capture(self, blob_name: str) -> 'GoogleStorageClient.GCSVideoCapture':
        """
        Get a GCSVideoCapture context manager for a video blob.
        
        Args:
            blob_name: Name of the video blob (will be prefixed with user_path)
            
        Returns:
            GCSVideoCapture: Context manager for video processing
            
        Raises:
            RuntimeError: If authentication fails or blob doesn't exist
        """
        if not self._ensure_authenticated():
            raise RuntimeError("Failed to authenticate with Google Cloud Storage")
        
        try:
            # Add user_path prefix to blob name
            if self.user_id and not self.user_id.endswith('/'):
                self.user_id += '/'
            if blob_name.startswith(self.user_id):
                full_blob_name = blob_name
            else:
                # Ensure user_id is prefixed correctly
                full_blob_name = f"{self.user_id}{blob_name}" 
            blob = self._bucket.blob(full_blob_name)
            
            # Check if blob exists
            if not blob.exists():
                raise FileNotFoundError(f"Video blob {full_blob_name} does not exist")
                
            return self.GCSVideoCapture(blob)
            
        except Exception as e:
            logger.error(f"Failed to create video capture for blob {blob_name}: {e}")
            raise
        

    class GCSVideoCapture:
        """
        A context manager for loading a video from GCS into memory and handling its lifecycle.
        Uses cv2 with an in-memory approach instead of downloading to disk.
        """
        def __init__(self, gcs_blob: Blob):
            self.gcs_blob = gcs_blob
            self.cap = None
            self.video_data = None
            self.temp_file_path = None

        def __enter__(self):
            """
            Executed when the 'with' block is entered.
            Downloads video data into memory and creates a temporary file for cv2.
            """
            logger.info(f"Initializing video capture for GCS blob: {self.gcs_blob.name}")

            try:
                # Download video data as bytes into memory
                self.video_data = self.gcs_blob.download_as_bytes()
                
                # Create a temporary file in memory and write the video data
                # Unfortunately, cv2.VideoCapture requires a file path, so we need a temporary file
                # But we'll use a more efficient approach with NamedTemporaryFile
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_file.write(self.video_data)
                temp_file.flush()
                self.temp_file_path = temp_file.name
                temp_file.close()

                # Open the video file with OpenCV
                self.cap = cv2.VideoCapture(self.temp_file_path)

                if not self.cap.isOpened():
                    raise IOError(f"Error: Could not open video file from GCS blob {self.gcs_blob.name}")
                return self

            except Exception as e:
                # Clean up on error
                if self.temp_file_path and os.path.exists(self.temp_file_path):
                    os.remove(self.temp_file_path)
                raise IOError(f"Failed to initialize video capture from GCS blob: {e}")

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Executed when the 'with' block is exited (even if an error occurred).
            Cleans up resources and removes temporary file.
            """
            if self.cap:
                self.cap.release()
            
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
                
            # Clear the video data from memory
            self.video_data = None
                
        def read(self):
            """
            Reads the next frame from the video.
            
            Returns:
                tuple: (ret, frame) where ret is boolean indicating success,
                       and frame is the video frame as numpy array
            """
            if self.cap:
                return self.cap.read()
            return False, None
        
        
        def get(self, prop_id):
            """
            Gets a property from the video capture.
            
            Args:
                prop_id: Property identifier (e.g., cv2.CAP_PROP_FRAME_COUNT)
                
            Returns:
                Property value
            """
            if self.cap:
                return self.cap.get(prop_id)
            return None
        
        def set(self, prop_id, value):
            """
            Sets a property of the video capture.
            
            Args:
                prop_id: Property identifier (e.g., cv2.CAP_PROP_POS_FRAMES)
                value: Value to set
                
            Returns:
                bool: True if successful, False otherwise
            """
            if self.cap:
                return self.cap.set(prop_id, value)
            return False
        
        def isOpened(self):
            """
            Checks if the video capture is opened.
            
            Returns:
                bool: True if opened, False otherwise
            """
            if self.cap:
                return self.cap.isOpened()
            return False
        
        def release(self):
            """
            Manually release the video capture (usually handled by __exit__).
            """
            if self.cap:
                self.cap.release()
                self.cap = None



def get_storage(*args) -> GoogleStorageClient:
    """
    Get a Google Storage client instance.
    
    Args:
        user_path: The user-specific path within the bucket (e.g., "tenant1/user123")
        credentials: Optional credentials for authentication

    Returns:
        GoogleStorageClient: Configured Google Storage client instance
    """

    return GoogleStorageClient(*args)




###########
# USAGE EXAMPLE

if __name__ == "__main__":
    # from common.google_storage import get_storage

    print("Google Cloud Storage Client Example")
    print("=" * 40)
    
    # Show .env file path being used
    print("-" * 40)
    
    # Example user path - caller must provide this
    user_path = "Common/Models"
    storage_client = get_storage(user_path)
    
    # Print configuration being used
    print(f"Project ID: {storage_client.config.project_id}")
    print(f"Bucket Name: {storage_client.config.bucket_name}")
    print(f"User Path: {storage_client.user_id}")
    print("-" * 40)

    try:
        # List all blobs in the bucket
        print("Listing blobs...")
        blobs = storage_client.list_blobs()
        
        if blobs:
            print(f"Found {len(blobs)} blobs:")
            for blob in blobs:
                print(f"  - {blob}")
        else:
            print("No blobs found in the specified path.")
            
        # Example of using GCSVideoCapture (uncomment to test with actual video)
        # video_blob_name = "raw/sample_video.mp4"
        # try:
        #     with storage_client.get_video_capture(video_blob_name) as video_cap:
        #         if video_cap.isOpened():
        #             frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #             fps = video_cap.get(cv2.CAP_PROP_FPS)
        #             print(f"Video loaded: {frame_count} frames at {fps} FPS")
        #             
        #             # Read first frame
        #             ret, frame = video_cap.read()
        #             if ret:
        #                 print(f"First frame shape: {frame.shape}")
        #         else:
        #             print("Failed to open video")
        # except Exception as video_error:
        #     print(f"Video processing error: {video_error}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your .env file is in the project root directory")
        print("2. Verify GOOGLE_APPLICATION_CREDENTIALS is set in the .env file")
        print("3. Check that the service account key file path is correct")
        print("4. Ensure the service account has access to the bucket")
        print("5. Verify the bucket exists and is accessible")