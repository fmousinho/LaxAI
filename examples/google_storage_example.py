"""
Example usage of the Google Storage client.

This example demonstrates how to use the GoogleStorageClient
for common operations like listing, uploading, and downloading files.
"""

import os
import tempfile
from core.common.google_storage import get_storage


def main():
    """Example usage of Google Storage client."""
    
    # Get storage client instance with user path
    user_path = "tenant1/user123"
    storage_client = get_storage(user_path)
    
    print(f"Using user path: {user_path}")
    
    try:
        # List all blobs in the bucket
        print("Listing all blobs in bucket:")
        blobs = storage_client.list_blobs()
        for blob_name in blobs:
            print(f"  - {blob_name}")
        
        # List blobs with prefix
        print("\nListing blobs with prefix 'data/':")
        data_blobs = storage_client.list_blobs(prefix="data/")
        for blob_name in data_blobs:
            print(f"  - {blob_name}")
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Hello, Google Cloud Storage!")
            tmp_file_path = tmp_file.name
        
        try:
            # Upload the file
            blob_name = "test_uploads/example.txt"
            print(f"\nUploading file to {blob_name}...")
            upload_success = storage_client.upload_blob(tmp_file_path, blob_name)
            
            if upload_success:
                print("Upload successful!")
                
                # Check if blob exists
                exists = storage_client.blob_exists(blob_name)
                print(f"Blob exists: {exists}")
                
                # Download the file
                download_path = tmp_file_path.replace('.txt', '_downloaded.txt')
                print(f"Downloading blob to {download_path}...")
                download_success = storage_client.download_blob(blob_name, download_path)
                
                if download_success:
                    print("Download successful!")
                    
                    # Read and display downloaded content
                    with open(download_path, 'r') as f:
                        content = f.read()
                        print(f"Downloaded content: {content}")
                    
                    # Clean up downloaded file
                    os.unlink(download_path)
                else:
                    print("Download failed!")
            else:
                print("Upload failed!")
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    except RuntimeError as e:
        print(f"Authentication error: {e}")
        print("Please ensure you have valid Google Cloud credentials configured.")
        print("You can set up credentials by:")
        print("1. Installing gcloud CLI and running 'gcloud auth application-default login'")
        print("2. Setting the GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("3. Running on Google Cloud Platform with service account")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
