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
from utils import load_env_or_colab
from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


@dataclass
class GoogleStorageConfig:
    project_id: str = "LaxAI"
    bucket_name: str = "laxai_dev"
    user_path: str = ""  # Will be set by caller
    credentials_name: str = "GOOGLE_APPLICATION_CREDENTIALS"

class GoogleStorageClient:
    """Google Cloud Storage client with error handling and common operations."""

    def __init__(self, user_path: str, credentials: Optional[service_account.Credentials]):
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
            # Add user_path prefix to search prefix, consistent with upload methods
            if prefix:
                if self.config.user_path:
                    full_prefix = f"{self.config.user_path}/{prefix}"
                else:
                    full_prefix = prefix
            else:
                full_prefix = self.config.user_path if self.config.user_path else None
                
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
            if len(self.config.user_path) > 0 and not self.config.user_path.endswith('/'):
                self.config.user_path += '/'
            full_destination = f"{self.config.user_path}{destination_blob_name}"
            blob = self._bucket.blob(full_destination)
            blob.upload_from_filename(source_file_path)
            logger.debug(f"File {source_file_path} uploaded to {full_destination}")
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
            # Get the blob object and download it
            # If the source_blob_name already starts with user_path, don't prefix it again
            if self.config.user_path and source_blob_name.startswith(self.config.user_path + "/"):
                full_source = source_blob_name
            else:
                full_source = f"{self.config.user_path}/{source_blob_name}" if self.config.user_path else source_blob_name
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
            if self.config.user_path and source_blob_name.startswith(self.config.user_path + "/"):
                full_source = source_blob_name
            else:
                full_source = f"{self.config.user_path}/{source_blob_name}" if self.config.user_path else source_blob_name
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


def get_storage(user_path: str, credentials: Optional[service_account.Credentials] = None) -> GoogleStorageClient:
    """
    Get a Google Storage client instance.
    
    Args:
        user_path: The user-specific path within the bucket (e.g., "tenant1/user123")
        credentials: Optional credentials for authentication

    Returns:
        GoogleStorageClient: Configured Google Storage client instance
    """

    return GoogleStorageClient(user_path, credentials)




###########
# USAGE EXAMPLE

if __name__ == "__main__":
    # from core.common.google_storage import get_storage

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
    print(f"User Path: {storage_client.config.user_path}")
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