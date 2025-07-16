from core.common.google_storage import get_storage

# Get a storage client instance with user path
user_path = "tenant1/user"  # Specify the user path
storage_client = get_storage(user_path)

