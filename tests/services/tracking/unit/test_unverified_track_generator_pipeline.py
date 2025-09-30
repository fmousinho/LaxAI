"""
Tests for TrackGeneratorPipeline class.

This module contains unit tests for the TrackGeneratorPipeline class,
focusing on initialization, configuration, and core functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
import supervision as sv

from services.service_tracking.src.unverified_track_generator_pipeline import TrackGeneratorPipeline
from config.all_config import DetectionConfig
from common.pipeline_step import StepStatus


class TestTrackGeneratorPipeline:
    """Test suite for TrackGeneratorPipeline."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock DetectionConfig for testing."""
        config = Mock(spec=DetectionConfig)
        config.model_path = "/fake/model/path"
        config.confidence_threshold = 0.5
        return config

    @pytest.fixture
    def mock_detection_model(self):
        """Create a mock DetectionModel."""
        model = Mock()
        model.generate_detections.return_value = sv.Detections.empty()
        return model

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock AffineAwareByteTrack tracker."""
        tracker = Mock()
        tracker.update_with_transform.return_value = sv.Detections.empty()
        tracker.get_identity_affine_matrix.return_value = np.eye(3)
        tracker.calculate_affine_transform.return_value = np.eye(3)
        return tracker

    @pytest.fixture
    def mock_storage(self):
        """Create a mock GoogleStorageClient."""
        storage = Mock()
        storage.move_blob.return_value = True
        storage.upload_from_bytes.return_value = True
        storage.list_blobs.return_value = []
        storage.get_video_capture.return_value = Mock()
        return storage

    @pytest.fixture
    def mock_path_manager(self):
        """Create a mock GCSPaths manager."""
        paths = Mock()
        paths.get_path.return_value = "fake/gcs/path"
        return paths

    @patch('services.service_tracking.src.unverified_track_generator_pipeline.get_storage')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.GCSPaths')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.DetectionModel')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.AffineAwareByteTrack')
    def test_initialization_success(self, mock_tracker_class, mock_detection_class,
                                   mock_paths_class, mock_storage_func,
                                   mock_config, mock_detection_model, mock_tracker,
                                   mock_storage, mock_path_manager):
        """Test successful pipeline initialization."""
        # Setup mocks
        mock_storage_func.return_value = mock_storage
        mock_paths_class.return_value = mock_path_manager
        mock_detection_class.return_value = mock_detection_model
        mock_tracker_class.return_value = mock_tracker

        # Initialize pipeline
        pipeline = TrackGeneratorPipeline(
            config=mock_config,
            tenant_id="test_tenant",
            verbose=True
        )

        # Verify initialization
        assert pipeline.config == mock_config
        assert pipeline.tenant_id == "test_tenant"
        assert pipeline.detection_model == mock_detection_model
        assert pipeline.tracker == mock_tracker
        assert pipeline.tenant_storage == mock_storage
        assert pipeline.path_manager == mock_path_manager

    @patch('services.service_tracking.src.unverified_track_generator_pipeline.get_storage')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.GCSPaths')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.DetectionModel')
    def test_initialization_detection_model_failure(self, mock_detection_class,
                                                   mock_paths_class, mock_storage_func,
                                                   mock_config):
        """Test initialization failure when detection model fails to load."""
        # Setup mocks
        mock_storage_func.return_value = Mock()
        mock_paths_class.return_value = Mock()
        mock_detection_class.side_effect = RuntimeError("Model load failed")

        # Verify exception is raised
        with pytest.raises(RuntimeError, match="Training pipeline cannot continue without detection model"):
            TrackGeneratorPipeline(
                config=mock_config,
                tenant_id="test_tenant"
            )

    @patch('services.service_tracking.src.unverified_track_generator_pipeline.get_storage')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.GCSPaths')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.DetectionModel')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.AffineAwareByteTrack')
    def test_initialization_tracker_failure(self, mock_tracker_class, mock_detection_class,
                                           mock_paths_class, mock_storage_func,
                                           mock_config, mock_detection_model):
        """Test initialization failure when tracker fails to initialize."""
        # Setup mocks
        mock_storage_func.return_value = Mock()
        mock_paths_class.return_value = Mock()
        mock_detection_class.return_value = mock_detection_model
        mock_tracker_class.side_effect = RuntimeError("Tracker init failed")

        # Verify exception is raised
        with pytest.raises(RuntimeError, match="Training pipeline cannot continue without tracker"):
            TrackGeneratorPipeline(
                config=mock_config,
                tenant_id="test_tenant"
            )

    @patch('unverified_track_generator_pipeline.get_storage')
    @patch('unverified_track_generator_pipeline.GCSPaths')
    @patch('unverified_track_generator_pipeline.DetectionModel')
    @patch('unverified_track_generator_pipeline.AffineAwareByteTrack')
    def test_run_with_empty_video_path(self, mock_tracker_class, mock_detection_class,
                                      mock_paths_class, mock_storage_func,
                                      mock_config, mock_detection_model, mock_tracker,
                                      mock_storage, mock_path_manager):
        """Test run method with empty video path."""
        # Setup mocks
        mock_storage_func.return_value = mock_storage
        mock_paths_class.return_value = mock_path_manager
        mock_detection_class.return_value = mock_detection_model
        mock_tracker_class.return_value = mock_tracker

        pipeline = TrackGeneratorPipeline(
            config=mock_config,
            tenant_id="test_tenant"
        )

        # Test empty video path
        result = pipeline.run("")
        assert result["status"] == "error"
        assert "resume_from_checkpoint is True but no video found in checkpoint context" in result["error"]

    @patch('services.service_tracking.src.unverified_track_generator_pipeline.get_storage')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.GCSPaths')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.DetectionModel')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.AffineAwareByteTrack')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.create_run_id')
    def test_run_calls_parent_pipeline(self, mock_create_run_id, mock_tracker_class,
                                      mock_detection_class, mock_paths_class,
                                      mock_storage_func, mock_config, mock_detection_model,
                                      mock_tracker, mock_storage, mock_path_manager):
        """Test that run method properly calls parent Pipeline.run."""
        # Setup mocks
        mock_storage_func.return_value = mock_storage
        mock_paths_class.return_value = mock_path_manager
        mock_detection_class.return_value = mock_detection_model
        mock_tracker_class.return_value = mock_tracker
        mock_create_run_id.return_value = "test_run_123"

        pipeline = TrackGeneratorPipeline(
            config=mock_config,
            tenant_id="test_tenant"
        )

        # Mock parent run method
        with patch.object(pipeline.__class__.__bases__[0], 'run') as mock_parent_run:
            mock_parent_run.return_value = {
                "status": "completed",
                "run_guid": "test_run_123",
                "run_folder": "test/folder",
                "context": {
                    "video_guid": "test_video_123",
                    "video_folder": "test/video/folder"
                },
                "errors": [],
                "pipeline_summary": {"total_detections": 10}
            }

            result = pipeline.run("test_video.mp4")

            # Verify parent run was called
            mock_parent_run.assert_called_once()
            assert result["status"] == "completed"
            assert result["video_guid"] == "test_video_123"

    @patch('services.service_tracking.src.unverified_track_generator_pipeline.get_storage')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.GCSPaths')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.DetectionModel')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.AffineAwareByteTrack')
    def test_stop_method(self, mock_tracker_class, mock_detection_class,
                        mock_paths_class, mock_storage_func,
                        mock_config, mock_detection_model, mock_tracker,
                        mock_storage, mock_path_manager):
        """Test stop method functionality."""
        # Setup mocks
        mock_storage_func.return_value = mock_storage
        mock_paths_class.return_value = mock_path_manager
        mock_detection_class.return_value = mock_detection_model
        mock_tracker_class.return_value = mock_tracker

        pipeline = TrackGeneratorPipeline(
            config=mock_config,
            tenant_id="test_tenant"
        )

        # Mock the stop_pipeline function
        with patch('shared_libs.common.pipeline.stop_pipeline') as mock_stop_pipeline:
            mock_stop_pipeline.return_value = True

            result = pipeline.stop()
            assert result is True
            mock_stop_pipeline.assert_called_once_with("unverified_tracks_pipeline")

    @patch('services.service_tracking.src.unverified_track_generator_pipeline.get_storage')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.GCSPaths')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.DetectionModel')
    @patch('services.service_tracking.src.unverified_track_generator_pipeline.AffineAwareByteTrack')
    def test_is_stopping_method(self, mock_tracker_class, mock_detection_class,
                               mock_paths_class, mock_storage_func,
                               mock_config, mock_detection_model, mock_tracker,
                               mock_storage, mock_path_manager):
        """Test is_stopping method."""
        # Setup mocks
        mock_storage_func.return_value = mock_storage
        mock_paths_class.return_value = mock_path_manager
        mock_detection_class.return_value = mock_detection_model
        mock_tracker_class.return_value = mock_tracker

        pipeline = TrackGeneratorPipeline(
            config=mock_config,
            tenant_id="test_tenant"
        )

        # Mock the is_stop_requested method
        with patch.object(pipeline, 'is_stop_requested', return_value=True) as mock_is_stop_requested:
            assert pipeline.is_stopping() is True
            mock_is_stop_requested.assert_called_once()

    def test_constants(self):
        """Test that pipeline constants are properly defined."""
        from services.service_tracking.src.unverified_track_generator_pipeline import (
            MIN_VIDEO_RESOLUTION,
            FRAME_SAMPLING_FOR_CROP,
            CROP_BATCH_SIZE,
            MAX_CONCURRENT_UPLOADS,
            CHECKPOINT_FRAME_INTERVAL
        )

        assert MIN_VIDEO_RESOLUTION == (1920, 1080)
        assert FRAME_SAMPLING_FOR_CROP == 15
        assert CROP_BATCH_SIZE == 5
        assert MAX_CONCURRENT_UPLOADS == 2
        assert CHECKPOINT_FRAME_INTERVAL == 100