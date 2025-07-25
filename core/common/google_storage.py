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
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)

# Load environment variables from .env file in the project root
# Get the directory containing this file (core/common/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach the project root
project_root = os.path.dirname(os.path.dirname(current_dir))
# Load .env file from project root
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)


@dataclass
class GoogleStorageConfig:
    project_id: str = "LaxAI"
    bucket_name: str = "laxai_dev"
    user_path: str = ""  # Will be set by caller
    credentials_path: Optional[str] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

class GoogleStorageClient:
    """Google Cloud Storage client with error handling and common operations."""
    
    def __init__(self, user_path: str):
        """
        Initialize the Google Storage client with predefined configuration.
        
        Args:
            user_path: The user-specific path within the bucket (e.g., "tenant1/user123")
        """
        self.config = GoogleStorageConfig()
        self.config.user_path = user_path
        self._client = None
        self._bucket = None
        self._authenticated = False
        
    def _authenticate(self) -> bool:
        """
        Authenticate with Google Cloud Storage.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # Set credentials path if provided in config (from environment variable)
            if self.config.credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.credentials_path
                logger.info(f"Using service account credentials from: {self.config.credentials_path}")
            
            # Create client with explicit project ID if provided
            if self.config.project_id:
                self._client = storage.Client(project=self.config.project_id)
                logger.info(f"Using explicitly configured project: {self.config.project_id}")
            else:
                # Use default project from ADC
                self._client = storage.Client()
                logger.info(f"Using project from ADC: {self._client.project}")

            # Test authentication by trying to get bucket
            self._bucket = self._client.bucket(self.config.bucket_name)
            
            # Test bucket access
            self._bucket.reload()
            
            self._authenticated = True
            logger.info(f"Successfully authenticated with Google Cloud Storage for bucket: {self.config.bucket_name}")
            return True
            
        except DefaultCredentialsError as e:
            logger.error(f"Authentication failed - No valid credentials found: {e}")
            logger.error("Make sure to set GOOGLE_APPLICATION_CREDENTIALS in your .env file or run 'gcloud auth application-default login'")
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
    
    def list_blobs(self, prefix: Optional[str] = None) -> List[str]:
        """
        Lists all the blobs in the bucket.
        
        Args:
            prefix: Optional prefix to filter blobs (will be combined with user_path)
            
        Returns:
            List of blob names
            
        Raises:
            RuntimeError: If authentication fails
        """
        if not self._ensure_authenticated():
            raise RuntimeError("Failed to authenticate with Google Cloud Storage")
        
        try:
            # Combine user_path with optional prefix
            full_prefix = self.config.user_path
            if prefix:
                full_prefix = f"{self.config.user_path}/{prefix}"
            
            blobs = self._client.list_blobs(self.config.bucket_name, prefix=full_prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list blobs: {e}")
            raise
    
    def upload_blob(self, source_file_path: str, destination_blob_name: str) -> bool:
        """
        Upload a file to the bucket.
        
        Args:
            source_file_path: Path to the source file
            destination_blob_name: Name of the blob in the bucket (will be prefixed with user_path)
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with Google Cloud Storage")
            return False
        
        try:
            # Add user_path prefix to destination
            full_destination = f"{self.config.user_path}/{destination_blob_name}"
            blob = self._bucket.blob(full_destination)
            blob.upload_from_filename(source_file_path)
            logger.info(f"File {source_file_path} uploaded to {full_destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
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
            # Add user_path prefix to source
            full_source = f"{self.config.user_path}/{source_blob_name}"
            blob = self._bucket.blob(full_source)
            blob.download_to_filename(destination_file_path)
            logger.info(f"Blob {full_source} downloaded to {destination_file_path}")
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
            # Add user_path prefix to source
            full_source = f"{self.config.user_path}/{source_blob_name}"
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
            full_destination = f"{self.config.user_path}/{destination_blob_name}"
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
        return self.upload_blob(file_path, destination_blob_name)
    
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
            full_blob_name = f"{self.config.user_path}/{blob_name}"
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
            full_blob_name = f"{self.config.user_path}/{blob_name}"
            blob = self._bucket.blob(full_blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check if blob exists: {e}")
            return False
    
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
            full_destination = f"{self.config.user_path}/{destination_blob_name}"
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
        return self.upload_blob(file_path, destination_blob_name)
    
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
            full_blob_name = f"{self.config.user_path}/{blob_name}"
            blob = self._bucket.blob(full_blob_name)
            blob.delete()
            logger.info(f"Blob {full_blob_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete blob: {e}")
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
            full_source = f"{self.config.user_path}/{source_blob_name}"
            full_destination = f"{self.config.user_path}/{destination_blob_name}"
            
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
            
            logger.info(f"Blob moved from {full_source} to {full_destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move blob: {e}")
            return False


def get_storage(user_path: str) -> GoogleStorageClient:
    """
    Get a Google Storage client instance.
    
    Args:
        user_path: The user-specific path within the bucket (e.g., "tenant1/user123")
    
    Returns:
        GoogleStorageClient: Configured Google Storage client instance
    """
    return GoogleStorageClient(user_path)




###########
# USAGE EXAMPLE

if __name__ == "__main__":
    # from core.common.google_storage import get_storage

    print("Google Cloud Storage Client Example")
    print("=" * 40)
    
    # Show .env file path being used
    print(f"Loading .env from: {env_path}")
    print(f".env file exists: {os.path.exists(env_path)}")
    print("-" * 40)
    
    # Example user path - caller must provide this
    user_path = "Common/Models"
    storage_client = get_storage(user_path)
    
    # Print configuration being used
    print(f"Project ID: {storage_client.config.project_id}")
    print(f"Bucket Name: {storage_client.config.bucket_name}")
    print(f"User Path: {storage_client.config.user_path}")
    print(f"Credentials Path: {storage_client.config.credentials_path}")
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
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your .env file is in the project root directory")
        print("2. Verify GOOGLE_APPLICATION_CREDENTIALS is set in the .env file")
        print("3. Check that the service account key file path is correct")
        print("4. Ensure the service account has access to the bucket")
        print("5. Verify the bucket exists and is accessible")