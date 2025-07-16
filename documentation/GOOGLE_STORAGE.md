# Google Storage Client

A robust Google Cloud Storage client with comprehensive error handling and authentication management.

## Features

- **Automatic Authentication**: Handles Google Cloud authentication with proper error handling
- **Comprehensive Error Handling**: Graceful handling of authentication failures, missing buckets, and permission issues
- **Caching**: Reuses authenticated connections for better performance
- **Common Operations**: Support for listing, uploading, downloading, and checking blob existence
- **Configurable**: Uses predefined project and bucket settings from configuration

## Configuration

The client uses a predefined configuration with the following defaults:

```python
@dataclass
class GoogleStorageConfig:
    project_id: str = "LaxAI"
    bucket_name: str = "laxai_dev"
    user_path: str = ""  # Set by caller when creating client
    credentials_path: Optional[str] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
```

The `user_path` must be provided by the caller when creating the client instance, enabling flexible tenant and user isolation.

## Usage

### Basic Usage

```python
from core.common.google_storage import get_storage

# Get a storage client instance with user-specific path
storage_client = get_storage("tenant1/user123")

# List all blobs in the user's path (tenant1/user123/)
blobs = storage_client.list_blobs()
for blob_name in blobs:
    print(blob_name)

# Upload a file (will be stored as tenant1/user123/remote_file.txt)
success = storage_client.upload_blob("local_file.txt", "remote_file.txt")
if success:
    print("Upload successful!")

# Download a file (downloads from tenant1/user123/remote_file.txt)
success = storage_client.download_blob("remote_file.txt", "downloaded_file.txt")
if success:
    print("Download successful!")

# Check if a blob exists (checks tenant1/user123/remote_file.txt)
exists = storage_client.blob_exists("remote_file.txt")
print(f"File exists: {exists}")
```

### Advanced Usage

```python
# Create clients for different users/tenants
user1_client = get_storage("tenant1/user123")
user2_client = get_storage("tenant2/user456")
admin_client = get_storage("admin/system")

# List blobs with additional prefix (will search in tenant1/user123/data/)
data_blobs = user1_client.list_blobs(prefix="data/")

# Handle authentication errors
try:
    blobs = user1_client.list_blobs()
except RuntimeError as e:
    print(f"Authentication failed: {e}")
```

## Environment Variables

The client supports configuration through environment variables in a `.env` file:

```bash
# .env file in project root
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Optional overrides (currently only GOOGLE_APPLICATION_CREDENTIALS is used)
# GCS_PROJECT_ID=custom-project
# GCS_BUCKET_NAME=custom-bucket
# GCS_USER_PATH=custom/user/path
```

The client automatically loads the `.env` file from the project root directory.

## Authentication

The client supports multiple authentication methods:

1. **Application Default Credentials** (recommended for local development):
   ```bash
   gcloud auth application-default login
   ```

2. **Service Account Key File**:
   Set the credentials path in your `.env` file:
   ```bash
   # In .env file
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

3. **Google Cloud Platform**: Automatically uses service account when running on GCP

## Error Handling

The client handles various error scenarios:

- **DefaultCredentialsError**: No valid credentials found
- **NotFound**: Bucket doesn't exist
- **Forbidden**: Access denied to bucket
- **General exceptions**: Other Google Cloud API errors

All methods return appropriate success/failure indicators and log errors for debugging.

## API Reference

### GoogleStorageClient

#### Methods

- `list_blobs(prefix: Optional[str] = None) -> List[str]`
  - Lists all blobs in the user's path, optionally filtered by additional prefix
  - The prefix is combined with the user_path (e.g., "Tenant1/User/prefix")
  - Returns list of blob names
  - Raises RuntimeError if authentication fails

- `upload_blob(source_file_path: str, destination_blob_name: str) -> bool`
  - Uploads a local file to the bucket in the user's path
  - Final path will be: user_path/destination_blob_name
  - Returns True if successful, False otherwise

- `download_blob(source_blob_name: str, destination_file_path: str) -> bool`
  - Downloads a blob from the user's path to a local file
  - Looks for: user_path/source_blob_name
  - Returns True if successful, False otherwise

- `blob_exists(blob_name: str) -> bool`
  - Checks if a blob exists in the user's path
  - Checks: user_path/blob_name
  - Returns True if exists, False otherwise

### get_storage(user_path: str)

Creates a new instance of `GoogleStorageClient` with the specified user path.

**Parameters:**
- `user_path` (str): The user-specific path within the bucket (e.g., "tenant1/user123", "admin/system")

**Returns:**
- `GoogleStorageClient`: Configured Google Storage client instance

**Example:**
```python
# Different user paths for different tenants/users
client1 = get_storage("tenant1/user123")
client2 = get_storage("tenant2/user456")
admin_client = get_storage("admin/system")
```

## Key Features

- **Flexible User Paths**: Caller specifies the user path, enabling multi-tenant architecture
- **Tenant Isolation**: All operations are automatically scoped to the specified user path
- **Automatic Path Handling**: You don't need to specify the full path, just the filename
- **Environment Configuration**: Loads credentials from `.env` file in project root
- **Comprehensive Error Handling**: Graceful handling of authentication and permission issues
- **Caching**: Reuses authenticated connections for better performance

## Example Usage Flow

```python
# Create client with specific user path
storage_client = get_storage("tenant1/user123")

# This uploads to: laxai_dev/tenant1/user123/my_model.pkl
storage_client.upload_blob("local_model.pkl", "my_model.pkl")

# This lists files in: laxai_dev/tenant1/user123/
files = storage_client.list_blobs()

# This lists files in: laxai_dev/tenant1/user123/models/
model_files = storage_client.list_blobs(prefix="models/")

# This downloads from: laxai_dev/tenant1/user123/my_model.pkl
storage_client.download_blob("my_model.pkl", "downloaded_model.pkl")

# Different user gets different path
other_client = get_storage("tenant2/user456")
# This uploads to: laxai_dev/tenant2/user456/my_model.pkl
other_client.upload_blob("local_model.pkl", "my_model.pkl")
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_google_storage.py -v
```

The tests use mocking to avoid actual Google Cloud API calls during testing.

## Example

See `examples/google_storage_example.py` for a complete working example.
