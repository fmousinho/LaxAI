


import os
import logging
import io
import time
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.auth.exceptions
from googleapiclient.errors import HttpError
from googleapiclient.discovery import Resource

logger = logging.getLogger(__name__)

# Directory where this script is located
_STORE_DRIVER_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for credentials and token files, relative to the script's location
_CREDENTIALS_PATH = os.path.join(_STORE_DRIVER_SCRIPT_DIR, "credentials.json")
_TOKEN_PATH = os.path.join(_STORE_DRIVER_SCRIPT_DIR, "token.json")
_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

#Cache directory to avoid downloading the same file multiple times
_CACHE_DIR = os.path.join(os.path.dirname(_STORE_DRIVER_SCRIPT_DIR), ".file_cache")
_CACHE_DURATION_SECONDS = 60 * 60 * 24 * 7 # Default 7 days

# GoogleDriveStore is used by Store, which is the main interface for the application.

class GoogleDriveStore:
    """
    Handles connection and interaction with Google Drive via the API.The only method
    that should be used to access files is `get_file_by_name`, which will handle
    authentication, caching, and downloading files as needed.
    This class uses a context manager to ensure proper resource management.
    A similar class could be used to handle other cloud storage services
    (e.g., AWS S3, Azure Blob Storage) by implementing the same interface.
    """
    def __init__(self):
        """
        Initializes the Store by authenticating with the Google Drive API.
        Paths for credentials, token, and cache_dir are determined internally,
        relative to this script's location, ensuring they are correctly resolved
        within the package structure.
        
        """
        self.credentials_path = _CREDENTIALS_PATH
        self.token_path = _TOKEN_PATH
        self.scopes = _SCOPES
        self.cache_dir = _CACHE_DIR              
        self.cache_duration = _CACHE_DURATION_SECONDS 

        #Establish the service object to interact with Google Drive API
        self.service = self._authenticate() 

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __enter__(self):
        """
        Context manager entry point.
        Returns the Store instance itself.
        """
        logger.debug("Entering Store context.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.
        Currently, no specific cleanup is needed for the Google Drive service object.
        """
        logger.debug("Exiting Store context.")
        # If there were resources to release (e.g., manually opened files or connections),
        # they would be handled here. The Google API client handles its own session.
        return False # Return False to propagate exceptions, True to suppress

    # --- Google Drive Authentication Helpers ---

    def _delete_token_file(self, reason: str = "problematic"):
        """Helper to delete the token file."""
        logger.info(f"Attempting to delete {reason} token file: {self.token_path}")
        try:
            if os.path.exists(self.token_path):
                os.remove(self.token_path)
                logger.info(f"Successfully deleted token file: {self.token_path}.")
            else:
                logger.info(f"Token file {self.token_path} not found, no need to delete.")
        except OSError as ose:
            logger.error(f"Failed to delete token file {self.token_path}: {ose}. Please delete it manually.")

    def _load_credentials_from_token(self) -> Optional[Credentials]:
        """Loads credentials from the token file if it exists."""
        if os.path.exists(self.token_path):
            try:
                return Credentials.from_authorized_user_file(self.token_path, self.scopes)
            except ValueError as ve: # Often indicates malformed JSON or token structure
                logger.error(f"Error loading token from {self.token_path} (ValueError: {ve}). File might be corrupted.")
                self._delete_token_file("corrupted")
            except Exception as e:
                logger.error(f"Unexpected error loading token from {self.token_path}: {e}")
        return None

    def _refresh_credentials(self, creds: Credentials) -> Optional[Credentials]:
        """Attempts to refresh expired credentials."""
        logger.info("Attempting to refresh expired token...")
        try:
            creds.refresh(Request())
            logger.info("Token refreshed successfully.")
            return creds
        except google.auth.exceptions.RefreshError as e:
            logger.error(f"Error refreshing token: {e}. This usually means the refresh token is invalid or revoked.")
            self._delete_token_file("invalid or revoked")
        except Exception as e:
            logger.error(f"An unexpected error occurred during token refresh: {e}. Please re-authenticate.")
        return None # Refresh failed

    def _perform_new_authentication(self) -> Optional[Credentials]:
        """Performs a new authentication flow via the browser."""
        logger.info("Attempting to authenticate via browser flow...")
        if not os.path.exists(self.credentials_path):
            logger.error(f"Credentials file not found at: {self.credentials_path}")
            logger.error("Please download 'credentials.json' from Google Cloud Console "
                         "and place it in the project directory.")
            return None
        try:
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.scopes)
            creds = flow.run_local_server(port=0)
            return creds
        except Exception as e:
            logger.error(f"Failed during new authentication flow: {e}", exc_info=True)
            return None

    def _save_credentials(self, creds: Credentials):
        """Saves the given credentials to the token file."""
        try:
            with open(self.token_path, 'w') as token_file:
                token_file.write(creds.to_json())
            logger.info(f"Credentials saved to {self.token_path}")
        except Exception as e:
            logger.error(f"Error saving token to {self.token_path}: {e}")

    def _authenticate(self) -> Optional[Resource]:
        """
        Orchestrates Google Drive API authentication.

        Returns:
            A Google Drive API service object (googleapiclient.discovery.Resource)
            or None if authentication fails.
        """
        creds = self._load_credentials_from_token()

        if creds and creds.valid:
            logger.info("Using valid credentials from token file.")
        else: # creds is None, or not valid
            if creds and creds.expired and creds.refresh_token: # Attempt refresh only if possible
                refreshed_creds = self._refresh_credentials(creds)
                if refreshed_creds and refreshed_creds.valid:
                    creds = refreshed_creds
                    self._save_credentials(creds) # Save the newly refreshed token
                else: # Refresh failed or resulted in invalid creds
                    logger.info("Token refresh failed or token still invalid.")
                    creds = None # Force new authentication
            else: # No creds, or creds exist but are invalid and cannot be refreshed
                logger.info("No valid credentials found or refresh not possible.")
                creds = None # Force new authentication

        if not creds or not creds.valid: # If still no valid creds after load/refresh attempts
            logger.info("Proceeding to new authentication.")
            creds = self._perform_new_authentication()
            if creds and creds.valid:
                self._save_credentials(creds)
            else:
                logger.error("Failed to obtain valid credentials after all authentication attempts.")
                return None # Critical failure to get any credentials

        # Build service if creds are now (hopefully) valid
        try:
            service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive API service created successfully.")
            return service
        except HttpError as error:
            logger.error(f'An API error occurred: {error}')
            return None
        except Exception as e:
            logger.error(f'An unexpected error occurred building the Drive service: {e}', exc_info=True)
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
                # Delete the expired file
                try:
                    os.remove(cache_filepath)
                except OSError as e:
                    logger.warning(f"Could not delete expired cache file {cache_filepath}: {e}")
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
        
    # --- File and Folder Management Methods ---

    def _find_file_id(self, file_name: str, parent_id: Optional[str] = None) -> Optional[str]:
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
        
    def _find_folder_id(self, folder_path: str, parent_id: str = 'root') -> Optional[str]:
        """
        Finds the Google Drive ID of a folder specified by a path string.

        Args:
            folder_path: A string representing the path relative to the parent_id
                         (e.g., 'MyProject/Data' or 'MyProject\\Data').
            parent_id: The ID of the parent folder to start searching from (defaults to 'root').

        Returns:
            An Optional ID of the target folder, or None if not found or an error occurs.
        """

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
    
    # --- External Methods ---

    def get_file_by_name(
            self, file_name: str,
            file_id: Optional[str] = None, 
            folder_path: Optional[str] = None, 
            parent_folder_id: Optional[str] = None
            ) -> Optional[io.BytesIO]:
        """
        Downloads a file based on its name and its location within a specified folder path or parent ID.
        If theres is conflict, the resolution order is as follows:
        1. If file_id is provided, it will be used directly to download the file.
        2. If folder_path is provided, it will be used to find the parent folder ID,
           and then the file will be searched within that folder.
        3. If neither is provided, the file will be searched globally in Drive.
        If the file is found, it will be downloaded and cached for future use.

        Args:
            file_name: The name of the file to download.
            file_id: Optional ID of the file to download directly.
            folder_path: A string representing the path from 'root' (e.g., 'MyProject/Data').
                         If provided, parent_folder_id is ignored for finding the folder,
                         but the search starts from 'root'.
            parent_folder_id: The direct ID of the parent folder containing the file.
                              Used if folder_path is None. Defaults to None (search anywhere).

        Returns:
            An Optional io.BytesIO object containing the file content, or None if not found
             or an error occurs.
        """
        actual_file_id_to_download = file_id  # Prioritize explicitly provided file_id

        if not actual_file_id_to_download:
            # If file_id is not provided, determine the parent folder to search in
            target_search_parent_id = parent_folder_id
            if folder_path:
                logger.info(f"Finding parent folder ID for path: '{folder_path}' to search for '{file_name}'.")
                resolved_folder_id = self._find_folder_id(folder_path)
                if not resolved_folder_id:
                    logger.error(f"Could not find the folder for path: '{folder_path}' while trying to locate '{file_name}'.")
                    return None
                target_search_parent_id = resolved_folder_id
            
            # Find the file ID by name within the determined parent folder (or globally)
            logger.info(f"Searching for file '{file_name}' in parent '{target_search_parent_id or 'Drive root/globally'}'.")
            actual_file_id_to_download = self._find_file_id(file_name, parent_id=target_search_parent_id)

            if not actual_file_id_to_download:
                # find_file_id already logs a warning if not found
                return None
        else:
            logger.info(f"Using provided file_id '{actual_file_id_to_download}' for file '{file_name}'.")

        # At this point, actual_file_id_to_download is the ID of the file we intend to download.
        # Cache identifier should be based on this definitive file ID.
        cache_identifier = actual_file_id_to_download 
        cache_filepath = self._get_cache_filepath(cache_identifier)

        # Check cache first
        if cache_filepath and self._is_cache_valid(cache_filepath):
            logger.info(f"Using cached content for file ID '{actual_file_id_to_download}' (name: '{file_name}') from: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                return io.BytesIO(f.read())

        # If not in cache or cache is invalid, download the file
        logger.info(f"Downloading file ID '{actual_file_id_to_download}' (name: '{file_name}').")
        buffer = self.get_file_by_id(actual_file_id_to_download)

        # Save to cache if download was successful
        if buffer:
            if cache_filepath:
                self._save_to_cache(cache_filepath, buffer)
                buffer.seek(0) # Reset buffer position after saving to cache for the caller
            return buffer
        
        # If buffer is None (download failed)
        logger.error(f"Failed to download file ID '{actual_file_id_to_download}' (name: '{file_name}').")
        return None
    
    def get_file_by_id(self, file_id: str) -> Optional[io.BytesIO]:
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
  
    def is_initialized(self) -> bool:
        """
        Checks if the Google Drive service was successfully initialized.
        Returns:
            True if the service is available, False otherwise.
        """
        return True if self.service else False
    
class Store(GoogleDriveStore):
    """
    A Store class that extends GoogleDriveStore to provide a more application-specific interface.
    """
    def __init__(self):
        """
        Initializes the Store by calling the parent class constructor
        """
        super().__init__()
