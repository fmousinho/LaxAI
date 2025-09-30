"""
Test for Google Cloud Storage video discovery functionality.

This test verifies that the GoogleStorageClient can be instantiated
and used to list videos from GCS.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

from shared_libs.common.google_storage import get_storage, GCSPaths

logger = logging.getLogger(__name__)


class TestGCSVideoDiscovery:
    """Test suite for GCS video discovery functionality."""

    @pytest.fixture
    def mock_tenant_id(self):
        """Mock tenant ID for testing."""
        return "test_tenant"

    @pytest.fixture
    def mock_video_blobs(self):
        """Mock video blob names returned from GCS."""
        return {
            "video1.mp4",
            "video2.avi",
            "video3.MOV",
            "video4.mkv",
            "not_a_video.txt",
            "another_file.jpg"
        }

    def test_instantiate_google_storage_client(self, mock_tenant_id):
        """Test that GoogleStorageClient can be instantiated via get_storage."""
        with patch('shared_libs.common.google_storage.GoogleStorageClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Instantiate the client
            storage_client = get_storage(mock_tenant_id)

            # Verify the client was created with correct tenant_id
            mock_client_class.assert_called_once_with(mock_tenant_id, credentials=None)
            assert storage_client is mock_client

    def test_list_videos_from_gcs(self, mock_tenant_id, mock_video_blobs):
        """Test listing and filtering videos from GCS."""
        with patch('shared_libs.common.google_storage.GoogleStorageClient') as mock_client_class, \
             patch('shared_libs.common.google_storage.GCSPaths') as mock_paths_class:

            # Setup mocks
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_paths_instance = MagicMock()
            mock_paths_class.return_value = mock_paths_instance
            mock_paths_instance.get_path.return_value = "raw/"

            # Configure list_blobs to return mock video blobs
            mock_client.list_blobs.return_value = mock_video_blobs

            # Instantiate components
            storage = get_storage(mock_tenant_id)
            path_manager = GCSPaths()

            # Get the raw videos path
            raw_videos_path = path_manager.get_path("raw_data")

            # List videos
            available_videos = storage.list_blobs(
                prefix=raw_videos_path,
                delimiter='/',
                exclude_prefix_in_return=True
            )

            # Verify list_blobs was called
            mock_client.list_blobs.assert_called_once()

            # Convert to list and filter for video files
            available_videos = list(available_videos)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            filtered_videos = [
                v for v in available_videos
                if any(v.lower().endswith(ext) for ext in video_extensions)
            ]

            # Verify filtering worked correctly
            expected_videos = ["video1.mp4", "video2.avi", "video3.MOV", "video4.mkv"]
            assert sorted(filtered_videos) == sorted(expected_videos)
            assert len(filtered_videos) == 4

    def test_video_discovery_integration(self, mock_tenant_id, caplog):
        """Integration test simulating the actual video discovery workflow."""
        with patch('shared_libs.common.google_storage.GoogleStorageClient') as mock_client_class, \
             patch('shared_libs.common.google_storage.GCSPaths') as mock_paths_class:

            # Setup mocks
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_paths_instance = MagicMock()
            mock_paths_class.return_value = mock_paths_instance
            mock_paths_instance.get_path.return_value = "test_tenant/raw_videos/"

            # Mock video blobs
            mock_video_blobs = {"video1.mp4", "video2.avi", "document.pdf"}
            mock_client.list_blobs.return_value = mock_video_blobs

            # Simulate the workflow from create_unverified_tracks.py
            storage = get_storage(mock_tenant_id)
            path_manager = GCSPaths()

            available_videos = []

            # Look for videos in the raw_videos directory
            raw_videos_path = path_manager.get_path("raw_data")

            with caplog.at_level(logging.INFO):
                try:
                    available_videos = storage.list_blobs(
                        prefix=raw_videos_path,
                        delimiter='/',
                        exclude_prefix_in_return=True
                    )
                    available_videos = list(available_videos)  # Convert to list

                    # Filter for video files
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
                    available_videos = [
                        v for v in available_videos
                        if any(v.lower().endswith(ext) for ext in video_extensions)
                    ]

                except Exception as e:
                    pytest.fail(f"Video discovery failed with error: {e}")

            # Verify results
            assert len(available_videos) == 2
            assert "video1.mp4" in available_videos
            assert "video2.avi" in available_videos
            assert "document.pdf" not in available_videos

    def test_empty_video_directory(self, mock_tenant_id):
        """Test behavior when no videos are found in GCS."""
        with patch('shared_libs.common.google_storage.GoogleStorageClient') as mock_client_class, \
             patch('shared_libs.common.google_storage.GCSPaths') as mock_paths_class:

            # Setup mocks
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_paths_instance = MagicMock()
            mock_paths_class.return_value = mock_paths_instance
            mock_paths_instance.get_path.return_value = "test_tenant/raw_videos/"

            # Mock empty result
            mock_client.list_blobs.return_value = set()

            # Test the workflow
            storage = get_storage(mock_tenant_id)
            path_manager = GCSPaths()

            raw_videos_path = path_manager.get_path("raw_data")

            available_videos = storage.list_blobs(
                prefix=raw_videos_path,
                delimiter='/',
                exclude_prefix_in_return=True
            )

            available_videos = list(available_videos)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            available_videos = [
                v for v in available_videos
                if any(v.lower().endswith(ext) for ext in video_extensions)
            ]

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires test MP4 video in GCS bucket")
    def test_gcs_connectivity_test_tenant_mp4_videos(self):
        """Integration test that connects to GCS and verifies test-tenant has exactly 1 mp4 video."""
        # This test makes real GCS calls - ensure credentials are available
        try:
            # Get real GCS storage client for test-tenant
            storage = get_storage("test-tenant")

            # Get path manager
            path_manager = GCSPaths()

            # Get the raw videos path
            raw_videos_path = path_manager.get_path("raw_data")

            # List videos in tenant1 raw data directory
            available_videos = storage.list_blobs(
                prefix=raw_videos_path
            )

            # Convert set to list and filter for video files
            available_videos = list(available_videos)
            logger.info(f"Found {len(available_videos)} total blobs in tenant1 raw data")
            for blob_name in available_videos:
                logger.info(f"Blob: {blob_name}")

            # Filter for video files (mp4, avi, mov, mkv, webm)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            video_files = [
                blob_name for blob_name in available_videos
                if any(blob_name.lower().endswith(ext) for ext in video_extensions)
            ]

            logger.info(f"Found {len(video_files)} video files: {video_files}")

            # Filter for mp4 files specifically
            mp4_videos = [
                name for name in video_files
                if name.lower().endswith('.mp4')
            ]

            logger.info(f"Found {len(mp4_videos)} mp4 videos: {mp4_videos}")

            # Assert exactly 1 mp4 video exists
            assert len(mp4_videos) == 1, f"Expected exactly 1 mp4 video, but found {len(mp4_videos)}: {mp4_videos}"

        except Exception as e:
            pytest.fail(f"GCS connectivity test failed: {e}")