import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from google.cloud.exceptions import NotFound, Forbidden
from google.auth.exceptions import DefaultCredentialsError

from core.common.google_storage import GoogleStorageClient, get_storage, GoogleStorageConfig


class TestGoogleStorageClient:
    """Test cases for GoogleStorageClient class."""
    
    def test_init(self):
        """Test client initialization."""
        with patch.dict(os.environ, {}, clear=True):
            client = GoogleStorageClient("test/user")
            assert client.config.project_id == "LaxAI"
            assert client.config.bucket_name == "laxai_dev"
            assert client.config.user_path == "test/user"
            assert client._client is None
            assert client._bucket is None
            assert client._authenticated is False
    
    @patch('core.common.google_storage.storage.Client')
    def test_authenticate_success(self, mock_storage_client):
        """Test successful authentication."""
        # Mock the storage client and bucket
        mock_client = Mock()
        mock_bucket = Mock()
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        
        with patch.dict(os.environ, {}, clear=True):
            client = GoogleStorageClient("test/user")
            result = client._authenticate()
        
        assert result is True
        assert client._authenticated is True
        assert client._client == mock_client
        assert client._bucket == mock_bucket
        mock_storage_client.assert_called_once_with(project="LaxAI")
        mock_client.bucket.assert_called_once_with("laxai_dev")
        mock_bucket.reload.assert_called_once()
    
    @patch('core.common.google_storage.storage.Client')
    def test_authenticate_credentials_error(self, mock_storage_client):
        """Test authentication failure due to credentials error."""
        mock_storage_client.side_effect = DefaultCredentialsError("No credentials")
        
        with patch.dict(os.environ, {}, clear=True):
            client = GoogleStorageClient("test/user")
            result = client._authenticate()
        
        assert result is False
        assert client._authenticated is False
        assert client._client is None
    
    @patch('core.common.google_storage.storage.Client')
    def test_authenticate_bucket_not_found(self, mock_storage_client):
        """Test authentication failure due to bucket not found."""
        mock_client = Mock()
        mock_storage_client.return_value = mock_client
        mock_client.bucket.side_effect = NotFound("Bucket not found")
        
        with patch.dict(os.environ, {}, clear=True):
            client = GoogleStorageClient("test/user")
            result = client._authenticate()
        
        assert result is False
        assert client._authenticated is False
    
    @patch('core.common.google_storage.storage.Client')
    def test_authenticate_forbidden(self, mock_storage_client):
        """Test authentication failure due to access denied."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.reload.side_effect = Forbidden("Access denied")
        
        with patch.dict(os.environ, {}, clear=True):
            client = GoogleStorageClient("test/user")
            result = client._authenticate()
        
        assert result is False
        assert client._authenticated is False
    
    @patch('core.common.google_storage.storage.Client')
    @patch.dict(os.environ, {}, clear=True)
    def test_authenticate_with_credentials_path(self, mock_storage_client):
        """Test authentication with custom credentials path."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        
        client = GoogleStorageClient("test/user")
        client.config.credentials_path = "/path/to/credentials.json"
        
        result = client._authenticate()
        
        assert result is True
        assert os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') == "/path/to/credentials.json"
    
    @patch('core.common.google_storage.storage.Client')
    def test_list_blobs_success(self, mock_storage_client):
        """Test successful blob listing."""
        # Setup mocks
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob1 = Mock()
        mock_blob1.name = "file1.txt"
        mock_blob2 = Mock()
        mock_blob2.name = "file2.txt"
        
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_client.list_blobs.return_value = [mock_blob1, mock_blob2]
        
        client = GoogleStorageClient("test/user")
        client._authenticate()  # Authenticate first
        
        result = client.list_blobs()
        
        assert result == ["file1.txt", "file2.txt"]
        mock_client.list_blobs.assert_called_once_with("laxai_dev", prefix="test/user")
    
    @patch('core.common.google_storage.storage.Client')
    def test_list_blobs_with_prefix(self, mock_storage_client):
        """Test blob listing with prefix."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_client.list_blobs.return_value = []
        
        client = GoogleStorageClient("test/user")
        client._authenticate()
        
        client.list_blobs(prefix="data/")
        
        mock_client.list_blobs.assert_called_once_with("laxai_dev", prefix="test/user/data/")
    
    @patch('core.common.google_storage.storage.Client')
    def test_list_blobs_authentication_failure(self, mock_storage_client):
        """Test list_blobs when authentication fails."""
        mock_storage_client.side_effect = DefaultCredentialsError("No credentials")
        
        client = GoogleStorageClient("test/user")
        
        with pytest.raises(RuntimeError, match="Failed to authenticate with Google Cloud Storage"):
            client.list_blobs()
    
    @patch('core.common.google_storage.storage.Client')
    def test_upload_blob_success(self, mock_storage_client):
        """Test successful blob upload."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        client = GoogleStorageClient("test/user")
        client._authenticate()
        
        result = client.upload_blob("local_file.txt", "remote_file.txt")
        
        assert result is True
        mock_bucket.blob.assert_called_once_with("test/user/remote_file.txt")
        mock_blob.upload_from_filename.assert_called_once_with("local_file.txt")
    
    @patch('core.common.google_storage.storage.Client')
    def test_upload_blob_authentication_failure(self, mock_storage_client):
        """Test upload_blob when authentication fails."""
        mock_storage_client.side_effect = DefaultCredentialsError("No credentials")
        
        client = GoogleStorageClient("test/user")
        
        result = client.upload_blob("local_file.txt", "remote_file.txt")
        
        assert result is False
    
    @patch('core.common.google_storage.storage.Client')
    def test_download_blob_success(self, mock_storage_client):
        """Test successful blob download."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        client = GoogleStorageClient("test/user")
        client._authenticate()
        
        result = client.download_blob("remote_file.txt", "local_file.txt")
        
        assert result is True
        mock_bucket.blob.assert_called_once_with("test/user/remote_file.txt")
        mock_blob.download_to_filename.assert_called_once_with("local_file.txt")
    
    @patch('core.common.google_storage.storage.Client')
    def test_download_blob_authentication_failure(self, mock_storage_client):
        """Test download_blob when authentication fails."""
        mock_storage_client.side_effect = DefaultCredentialsError("No credentials")
        
        client = GoogleStorageClient("test/user")
        
        result = client.download_blob("remote_file.txt", "local_file.txt")
        
        assert result is False
    
    @patch('core.common.google_storage.storage.Client')
    def test_blob_exists_success(self, mock_storage_client):
        """Test successful blob existence check."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_blob.exists.return_value = True
        
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        client = GoogleStorageClient("test/user")
        client._authenticate()
        
        result = client.blob_exists("test_file.txt")
        
        assert result is True
        mock_bucket.blob.assert_called_once_with("test/user/test_file.txt")
        mock_blob.exists.assert_called_once()
    
    @patch('core.common.google_storage.storage.Client')
    def test_blob_exists_authentication_failure(self, mock_storage_client):
        """Test blob_exists when authentication fails."""
        mock_storage_client.side_effect = DefaultCredentialsError("No credentials")
        
        client = GoogleStorageClient("test/user")
        
        result = client.blob_exists("test_file.txt")
        
        assert result is False
    
    @patch('core.common.google_storage.storage.Client')
    def test_ensure_authenticated_cached(self, mock_storage_client):
        """Test that _ensure_authenticated uses cached authentication."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        
        client = GoogleStorageClient("test/user")
        
        # First call should authenticate
        result1 = client._ensure_authenticated()
        assert result1 is True
        
        # Second call should use cached authentication
        result2 = client._ensure_authenticated()
        assert result2 is True
        
        # Should only call authentication once
        mock_storage_client.assert_called_once()


class TestGetStorage:
    """Test cases for get_storage function."""
    
    def test_get_storage_returns_client(self):
        """Test that get_storage returns a GoogleStorageClient instance."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_storage("test/user")
            assert isinstance(client, GoogleStorageClient)
            assert client.config.project_id == "LaxAI"
            assert client.config.bucket_name == "laxai_dev"
            assert client.config.user_path == "test/user"
    
    def test_get_storage_returns_new_instance(self):
        """Test that get_storage returns a new instance each time."""
        client1 = get_storage("user1/path")
        client2 = get_storage("user2/path")
        assert client1 is not client2
        assert client1.config.user_path == "user1/path"
        assert client2.config.user_path == "user2/path"


class TestGoogleStorageConfig:
    """Test cases for GoogleStorageConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        # Clear environment but keep the actual .env file loaded values
        config = GoogleStorageConfig()
        assert config.project_id == "LaxAI"
        assert config.bucket_name == "laxai_dev"
        assert config.user_path == ""  # Default is empty, set by caller
        # credentials_path will be whatever is in the .env file or None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GoogleStorageConfig(
            project_id="custom-project",
            bucket_name="custom-bucket",
            user_path="custom/path",
            credentials_path="/path/to/creds.json"
        )
        assert config.project_id == "custom-project"
        assert config.bucket_name == "custom-bucket"
        assert config.user_path == "custom/path"
        assert config.credentials_path == "/path/to/creds.json"
    
    def test_environment_variable_config(self):
        """Test configuration loading from environment variables."""
        test_env = {
            "GOOGLE_APPLICATION_CREDENTIALS": "/env/path/to/creds.json"
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            config = GoogleStorageConfig()
            # The hardcoded values will still be used since they're not using os.environ.get
            assert config.project_id == "LaxAI"
            assert config.bucket_name == "laxai_dev"
            assert config.user_path == ""  # Default is empty, set by caller
            assert config.credentials_path == "/env/path/to/creds.json"
    
    def test_credentials_path_from_environment(self):
        """Test that credentials_path is loaded from environment variable."""
        test_env = {
            "GOOGLE_APPLICATION_CREDENTIALS": "/test/path/to/creds.json"
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            config = GoogleStorageConfig()
            assert config.credentials_path == "/test/path/to/creds.json"


if __name__ == "__main__":
    pytest.main([__file__])
