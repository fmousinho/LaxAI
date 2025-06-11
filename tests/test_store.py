import unittest
import os
import sys
from unittest.mock import patch # Keep patch for module-level constants
import io
import tempfile
import shutil
import time

# Determine the project root directory to allow absolute imports
# __file__ is /Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/tests/test_store.py
# laxai_package_dir is /Users/fernandomousinho/Documents/Learning_to_Code/LaxAI
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_LAXAI_PACKAGE_DIR = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, '..'))
_PROJECT_ROOT_DIR = os.path.abspath(os.path.join(_LAXAI_PACKAGE_DIR, '..'))

if _PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_DIR)

from LaxAI.tools.store_driver import Store # type: ignore

# --- Constants for Integration Tests ---
# IMPORTANT: Adjust these to match a real file and folder on your Google Drive
TEST_EXISTING_FILE_NAME = "LaxAI_test_file.txt"
TEST_EXISTING_FOLDER_PATH = "Colab_Notebooks" # Path from Drive root, adjust if your test file is elsewhere
TEST_EXPECTED_FILE_CONTENT = b"This is a test file for LaxAI integration testing." # Expected content as bytes
TEST_NON_EXISTENT_FILE_NAME = "this_file_should_absolutely_not_exist_blah_blah.txt"

class TestStoreGetFileByName(unittest.TestCase):

    def setUp(self):
        """Set up for each test method."""
        # With the sys.path modification above, Store.__module__ should consistently be 'LaxAI.tools.store_driver'
        self.store_module_name = Store.__module__
        # The patches below rely on self.store_module_name being correctly identified.
        # If it's not 'LaxAI.tools.store_driver', the patches might fail.

        # 1. Create a temporary directory for the cache for this test run
        self.test_cache_dir = tempfile.mkdtemp()

        # 2. We are NOT mocking _authenticate anymore for integration tests.
        #    Real authentication will be attempted.
        # 3. Patch the module-level _CACHE_DIR to use our temporary directory
        self.patch_cache_dir_const = patch(f'{self.store_module_name}._CACHE_DIR', self.test_cache_dir)
        self.mock_cache_dir_const = self.patch_cache_dir_const.start()

        # 4. Patch cache duration for predictable cache expiry if needed for other tests
        self.patch_cache_duration_const = patch(f'{self.store_module_name}._CACHE_DURATION_SECONDS', 300) # 5 minutes
        self.mock_cache_duration_const = self.patch_cache_duration_const.start()

        # 5. Instantiate the Store. It will use the mocked _authenticate and _CACHE_DIR.
        self.store = Store()
        # Ensure credentials.json is in LaxAI/tools/ for authentication to work.

    def tearDown(self):
        """Clean up after each test method."""
        # self.patch_authenticate.stop() # No longer patching _authenticate
        self.patch_cache_dir_const.stop()
        self.patch_cache_duration_const.stop()
        shutil.rmtree(self.test_cache_dir) # Remove the temporary cache directory

    def test_get_existing_file_from_drive_and_cache(self):
        """
        Tests downloading an existing file from Google Drive, then retrieving it from cache.
        Requires TEST_EXISTING_FILE_NAME and TEST_EXISTING_FOLDER_PATH to be set up on Drive.
        """
        if not self.store.is_initialized():
            self.fail("Store service failed to initialize. Check credentials and network.")

        # --- First call: Download from Google Drive and cache ---
        print(f"\nAttempting to download: {TEST_EXISTING_FILE_NAME} from folder: {TEST_EXISTING_FOLDER_PATH}")
        buffer1 = self.store.get_file_by_name(
            file_name=TEST_EXISTING_FILE_NAME,
            folder_path=TEST_EXISTING_FOLDER_PATH
        )

        self.assertIsNotNone(buffer1, "Failed to download the existing file from Google Drive.")
        content1 = buffer1.read()
        self.assertEqual(content1, TEST_EXPECTED_FILE_CONTENT, "Downloaded content does not match expected content.")

        # Determine the expected cache file path (based on file ID, which we don't know beforehand in this test)
        # So, we find the file ID first, then check cache.
        # This part implicitly tests _find_file_id as well.
        file_id_for_cache = self.store._find_file_id(TEST_EXISTING_FILE_NAME, parent_id=self.store._find_folder_id(TEST_EXISTING_FOLDER_PATH))
        self.assertIsNotNone(file_id_for_cache, "Could not find the file ID for the test file on Drive.")
        
        cached_file_path = os.path.join(self.test_cache_dir, file_id_for_cache)
        self.assertTrue(os.path.exists(cached_file_path), "File was not found in the cache after download.")
        with open(cached_file_path, 'rb') as f_cache:
            self.assertEqual(f_cache.read(), TEST_EXPECTED_FILE_CONTENT, "Cached content does not match expected content.")

        # --- Second call: Retrieve from cache ---
        print(f"Attempting to retrieve from cache: {TEST_EXISTING_FILE_NAME}")
        buffer2 = self.store.get_file_by_name(
            file_name=TEST_EXISTING_FILE_NAME,
            folder_path=TEST_EXISTING_FOLDER_PATH
        )
        self.assertIsNotNone(buffer2, "Failed to retrieve the file from cache.")
        content2 = buffer2.read()
        self.assertEqual(content2, TEST_EXPECTED_FILE_CONTENT, "Cached content (second call) does not match expected content.")

    def test_get_non_existent_file_from_drive(self):
        """Tests attempting to download a non-existent file from Google Drive."""
        if not self.store.is_initialized():
            self.fail("Store service failed to initialize. Check credentials and network.")

        print(f"\nAttempting to download non-existent file: {TEST_NON_EXISTENT_FILE_NAME}")
        buffer = self.store.get_file_by_name(TEST_NON_EXISTENT_FILE_NAME)
        self.assertIsNone(buffer, "Expected None when trying to download a non-existent file.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
