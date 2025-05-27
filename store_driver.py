
import os
import logging
import io
import tempfile # Import tempfile module
import time # Import time for cache expiration
from typing import Optional, Callable # Import Optional and Callable
# Define constants locally to avoid licensing issues

# Determine the directory of the current script (store_driver.py)
# This will be the base for resolving relative paths for credentials, token, and default cache.
_STORE_DRIVER_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define constants locally, making them absolute paths relative to this script's location
CREDENTIALS_PATH = os.path.join(_STORE_DRIVER_SCRIPT_DIR, "credentials.json")
TOKEN_PATH = os.path.join(_STORE_DRIVER_SCRIPT_DIR, "token.json")
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
# Default cache_dir is also relative to this script's location (project root)
# This ensures that if Store is instantiated without cache_dir, it still resolves correctly.
CACHE_DIR = os.path.join(_STORE_DRIVER_SCRIPT_DIR, ".file_cache")
CACHE_DURATION_SECONDS = 60 * 60 * 24 * 7 # Default 7 days
# Remove the import for constants

# Imports for Google Drive API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.auth.exceptions # Import for RefreshError
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

class Store:
    """
    Handles connection and interaction with Google Drive via the API.
    """
    def __init__(
        self,
        credentials_path=CREDENTIALS_PATH,
        token_path=TOKEN_PATH,
        scopes=SCOPES,
        temp_file_registrar: Optional[Callable[[str], None]] = None,
        cache_dir: str = CACHE_DIR,
        cache_duration: int = CACHE_DURATION_SECONDS):
        """
        Initializes the store by authenticating with the Google Drive API. Paths for
        credentials, token, and cache_dir default to absolute paths within the
        application's directory.
        If relative paths are explicitly passed for these arguments, they will be
        resolved relative to the Current Working Directory.
        Args:
            credentials_path: Path to the OAuth 2.0 client secrets file.
            token_path: Path to store/retrieve the user's access token.
            scopes: List of OAuth scopes required. Defaults to const.SCOPES.
            temp_file_registrar: An optional function to call with the path of any created temp file.
            cache_dir: Directory to store cached files.
            cache_duration: Maximum age of cached files in seconds.
        """
        self.credentials_path = credentials_path # Will be absolute if default is used
        self.token_path = token_path             # Will be absolute if default is used
        self.scopes = scopes if scopes else SCOPES # Use provided scopes or default from constants
        self.temp_file_registrar = temp_file_registrar # Store the registrar function
        self.cache_dir = cache_dir               # Will be absolute if default or const.CACHE_DIR is used
        self.cache_duration = cache_duration
        self.service = self._authenticate()

        # Ensure cache directory exists
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _authenticate(self):
        """
        Handles Google Drive API authentication and returns a service object.

        Requires 'credentials.json' from Google Cloud Console for the first run.
        Stores obtained token in 'token.json' for subsequent runs.

        Returns:
            A Google Drive API service object (googleapiclient.discovery.Resource)
            or None if authentication fails.
        """
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first time.
        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
            except Exception as e:
                 logger.error(f"Error loading token from {self.token_path}: {e}")
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Attempting to refresh expired token...")
                try:
                    creds.refresh(Request())
                    logger.info("Token refreshed successfully.")
                except google.auth.exceptions.RefreshError as e:
                    logger.error(f"Error refreshing token: {e}. This usually means the refresh token is invalid or revoked.")
                    logger.info(f"Attempting to delete problematic token file: {self.token_path}")
                    try:
                        if os.path.exists(self.token_path):
                            os.remove(self.token_path)
                            logger.info(f"Successfully deleted token file: {self.token_path}. Please re-run to authenticate.")
                        else:
                            logger.info(f"Token file {self.token_path} not found, no need to delete.")
                    except OSError as ose:
                        logger.error(f"Failed to delete token file {self.token_path}: {ose}. Please delete it manually.")
                    creds = None # Force re-authentication by falling through
                except Exception as e: # Catch other potential errors during refresh
                    logger.error(f"An unexpected error occurred during token refresh: {e}. Please re-authenticate.")
                    creds = None # Force re-authentication by falling through
            # This block is now for when creds is None initially, or became None after a failed refresh, or was never valid
            if not creds or not creds.valid: # Re-check creds as it might have been set to None above
                logger.info("No valid credentials, attempting to authenticate via browser flow...")
                if not os.path.exists(self.credentials_path):
                    logger.error(f"Credentials file not found at: {self.credentials_path}")
                    logger.error("Please download 'credentials.json' from Google Cloud Console "
                                 "and place it in the project directory.")
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.scopes)
                # Run local server flow which will open a browser window for auth
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run, only if creds is not None
            if creds:
                try:
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                    logger.info(f"Credentials saved to {self.token_path}")
                except Exception as e:
                    logger.error(f"Error saving token to {self.token_path}: {e}")
            else: # If creds is still None after all attempts
                logger.error("Failed to obtain valid credentials after all attempts.")
                return None
        try:
            service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive API service created successfully.")
            return service
        except HttpError as error:
            logger.error(f'An API error occurred: {error}')
            return None
        except Exception as e:
            logger.error(f'An unexpected error occurred building the Drive service: {e}')
            return None

    # --- Caching Helpers ---

    def _get_cache_filepath(self, unique_file_identifier: str) -> Optional[str]:
        """Constructs the full path for a potential cached file."""
        if not self.cache_dir:
            return None
        # Use the identifier directly as the filename in the cache
        # Consider adding hashing if identifiers might contain invalid chars
        return os.path.join(self.cache_dir, unique_file_identifier)

    def _is_cache_valid(self, cache_filepath: str) -> bool:
        """Checks if a cached file exists and is within the cache duration."""
        if not cache_filepath or not os.path.exists(cache_filepath):
            return False
        try:
            file_age = time.time() - os.path.getmtime(cache_filepath)
            if file_age < self.cache_duration:
                return True
            else:
                logger.info(f"Cache file expired: {cache_filepath}")
                # Optionally delete the expired file here
                # try:
                #     os.remove(cache_filepath)
                # except OSError as e:
                #     logger.warning(f"Could not delete expired cache file {cache_filepath}: {e}")
                return False
        except OSError as e:
            logger.warning(f"Error checking cache file status {cache_filepath}: {e}")
            return False

    def _save_to_cache(self, cache_filepath: str, buffer: io.BytesIO):
        """Saves the content of a buffer to the cache file."""
        if cache_filepath:
            with open(cache_filepath, 'wb') as f:
                f.write(buffer.getvalue())
            logger.info(f"Saved downloaded content to cache: {cache_filepath}")
        
    def find_file_id(self, file_name: str, parent_id: str = None) -> Optional[str]:
        """Finds the ID of a file by name within an optional parent folder."""
        if not self.service:
            logger.error("Drive service not available.")
            return None

        query = f"name='{file_name}' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        try:
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                pageSize=1 # Expecting only one file with this name (in this parent)
            ).execute()
            items = results.get('files', [])
            if items:
                file_id = items[0].get('id')
                logger.info(f"Found file '{file_name}' with ID: {file_id}")
                return file_id
            else:
                logger.warning(f"File '{file_name}' not found" + (f" in parent '{parent_id}'." if parent_id else "."))
                return None
        except HttpError as error:
            logger.error(f"An API error occurred while searching for file '{file_name}': {error}")
            return None

    def download_file_by_id(self, file_id: str) -> Optional[io.BytesIO]:
        """
        Downloads a file's content given its ID into an in-memory buffer.

        Returns:
            An Optional io.BytesIO object containing the file content, or None on error.
        """
        if not self.service:
            logger.error("Drive service not available.")
            return None
        try:
            request = self.service.files().get_media(fileId=file_id)
            # Use io.BytesIO to handle the downloaded content in memory
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                logger.debug(f"Download {int(status.progress() * 100)}%.")
            logger.info(f"Successfully downloaded file ID: {file_id}")
            fh.seek(0) # Reset buffer position to the beginning for reading
            return fh # Return the BytesIO object
        except HttpError as error:
            logger.error(f"An API error occurred while downloading file ID '{file_id}': {error}")
            return None
        
    def find_folder_id(self, folder_path: str, parent_id: str = 'root') -> Optional[str]:
        """
        Finds the Google Drive ID of a folder specified by a path string.

        Args:
            folder_path: A string representing the path relative to the parent_id
                         (e.g., 'MyProject/Data' or 'MyProject\\Data').
            parent_id: The ID of the parent folder to start searching from (defaults to 'root').

        Returns:
            An Optional ID of the target folder, or None if not found or an error occurs.
        """
        if not self.service:
            logger.error("Drive service not available.")
            return None

        # Normalize path separators and split into components
        normalized_path = folder_path.replace('\\', '/')
        folder_path_components = [comp for comp in normalized_path.split('/') if comp]

        if not folder_path_components:
            logger.warning("Empty folder path provided.")
            return parent_id # Return the starting parent if path is empty

        current_parent_id = parent_id
        target_id = None

        try:
            for folder_name in folder_path_components:
                target_id = None # Reset for the current component
                query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{current_parent_id}' in parents and trashed=false"

                results = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='files(id, name)',
                    pageSize=1 # We only expect one folder with this name in the parent
                ).execute()
                items = results.get('files', [])

                if items:
                    target_id = items[0].get('id')
                    current_parent_id = target_id # Set for the next iteration
                else:
                    logger.warning(f"Folder component '{folder_name}' not found inside parent ID '{current_parent_id}'.")
                    return None # Folder component not found

            return target_id # Return the ID of the last component found
        except HttpError as error:
            logger.error(f"An API error occurred while searching for folder path '{folder_path}': {error}")
            return None
        
    def download_file_by_name(self, file_name: str, folder_path: str = None, parent_folder_id: str = None) -> Optional[io.BytesIO]:
        """
        Downloads a file based on its name and its location within a specified folder path or parent ID.

        Args:
            file_name: The name of the file to download.
            folder_path: A string representing the path from 'root' (e.g., 'MyProject/Data').
                         If provided, parent_folder_id is ignored for finding the folder,
                         but the search starts from 'root'.
            parent_folder_id: The direct ID of the parent folder containing the file.
                              Used if folder_path is None. Defaults to None (search anywhere).

        Returns:
            An Optional io.BytesIO object containing the file content, or None if not found
             or an error occurs.

        Note: This method now incorporates caching.
        """
        target_parent_id = parent_folder_id
        # Create a unique identifier for caching based on name and location
        cache_identifier = f"{target_parent_id or 'global'}_{file_name}"
        cache_filepath = self._get_cache_filepath(cache_identifier)

        # Check cache first
        if cache_filepath and self._is_cache_valid(cache_filepath):
            logger.info(f"Using cached content for '{file_name}' from: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                return io.BytesIO(f.read())

        # If path components are given, find the target folder ID first
        if folder_path:
            logger.info(f"Finding parent folder ID for path: '{folder_path}'")
            target_parent_id = self.find_folder_id(folder_path) # Search starts from 'root' by default in find_folder_id
            if not target_parent_id:
                logger.error(f"Could not find the folder for path: '{folder_path}'")
                return None # Stop if the target folder wasn't found

        # Find the file ID within the determined parent folder (or globally if no parent specified)
        file_id = self.find_file_id(file_name, parent_id=target_parent_id)

        # If file ID is found, download it
        if file_id:
            buffer = self.download_file_by_id(file_id)
            # Save to cache if download was successful
            if buffer and cache_filepath:
                self._save_to_cache(cache_filepath, buffer)
                buffer.seek(0) # Reset buffer position after saving
            return buffer
        else:
            # find_file_id already logs a warning if not found
            return None

    def download_file_to_temp(self, file_name: str, folder_path: str = None, parent_folder_id: str = None, suffix: str = None) -> Optional[str]:
        """
        Downloads a file by name (within an optional folder) and saves it to a temporary file.

        Args:
            file_name: The name of the file to download.
            folder_path: Optional path to the folder containing the file.
            parent_folder_id: Optional direct ID of the parent folder.
            suffix: Optional suffix for the temporary file (e.g., '.mp4').

        Returns:
            The absolute path to the created temporary file, or None if download or saving fails.

        Note: This method now incorporates caching. It will return a path to the
              cached file if valid, otherwise it downloads, caches the content,
              saves to a *new* temporary file, and returns the temporary path.
        """
        logger.info(f"Attempting to download '{file_name}' to a temporary file...")
        # Create a unique identifier for caching based on name and location
        cache_identifier = f"{parent_folder_id or folder_path or 'global'}_{file_name}" # Use path if parent_id unknown
        cache_filepath = self._get_cache_filepath(cache_identifier)

        # Check cache first - if valid, return the CACHED path directly
        if cache_filepath and self._is_cache_valid(cache_filepath):
            logger.info(f"Using cached file for '{file_name}': {cache_filepath}")
            return cache_filepath # Return the path to the persistent cache file

        file_buffer = self.download_file_by_name(file_name, folder_path, parent_folder_id)

        if file_buffer:
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    temp_file.write(file_buffer.getvalue())
                    temp_file_path = temp_file.name # Get the path
                # Register the file for cleanup if a registrar was provided
                if self.temp_file_registrar:
                    self.temp_file_registrar(temp_file_path)
                logger.info(f"File '{file_name}' successfully saved to temporary path: {temp_file_path}")
                return temp_file_path
            except Exception as e:
                logger.error(f"Error saving downloaded content of '{file_name}' to temporary file: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Failed to download '{file_name}', cannot save to temporary file.")
            return None
