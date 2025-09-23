"""
Tests for TrackGeneratorPipeline methods.

This module contains unit tests for individual methods of the TrackGeneratorPipeline class,
focusing on internal functionality and edge cases.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np
import cv2
import supervision as sv

from src.unverified_track_generator_pipeline import TrackGeneratorPipeline
from config.all_config import DetectionConfig
from common.pipeline_step import StepStatus


class TestTrackGeneratorPipelineMethods:
    """Test suite for TrackGeneratorPipeline methods."""

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
        
        # Create a context manager mock for video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.__enter__ = Mock(return_value=mock_cap)
        mock_cap.__exit__ = Mock(return_value=None)
        
        storage.get_video_capture.return_value = mock_cap
        return storage

    @pytest.fixture
    def mock_path_manager(self):
        """Create a mock GCSPaths manager."""
        paths = Mock()
        paths.get_path.return_value = "fake/gcs/path"
        return paths

    @pytest.fixture
    def pipeline(self, mock_config, mock_detection_model, mock_tracker,
                mock_storage, mock_path_manager):
        """Create a TrackGeneratorPipeline instance with mocked dependencies."""
        with patch('unverified_track_generator_pipeline.get_storage', return_value=mock_storage), \
             patch('unverified_track_generator_pipeline.GCSPaths', return_value=mock_path_manager), \
             patch('unverified_track_generator_pipeline.DetectionModel', return_value=mock_detection_model), \
             patch('unverified_track_generator_pipeline.AffineAwareByteTrack', return_value=mock_tracker), \
             patch('unverified_track_generator_pipeline.training_config') as mock_training_config:

            mock_training_config.num_workers = 4  # Set a reasonable value for testing

            pipeline = TrackGeneratorPipeline(
                config=mock_config,
                tenant_id="test_tenant",
                verbose=False
            )
            return pipeline

    def test_validate_video_resolution_valid(self, pipeline):
        """Test video resolution validation with valid resolution."""
        assert pipeline._validate_video_resolution(1920, 1080) is True
        assert pipeline._validate_video_resolution(2560, 1440) is True

    def test_validate_video_resolution_invalid(self, pipeline):
        """Test video resolution validation with invalid resolution."""
        assert pipeline._validate_video_resolution(1280, 720) is False
        assert pipeline._validate_video_resolution(1920, 1079) is False
        assert pipeline._validate_video_resolution(1919, 1080) is False

    @patch('unverified_track_generator_pipeline.create_video_id')
    def test_import_video_success(self, mock_create_video_id, pipeline, mock_storage, mock_path_manager):
        """Test successful video import."""
        # Setup mocks
        mock_create_video_id.return_value = "test_video_123"
        mock_path_manager.get_path.return_value = "tenant/test_tenant/imported_videos/test_video_123"

        # Mock video capture for resolution validation
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.__enter__ = Mock(return_value=mock_cap)
        mock_cap.__exit__ = Mock(return_value=None)
        mock_storage.get_video_capture.return_value = mock_cap

        context = {"raw_video_path": "raw_videos/test.mp4"}

        result = pipeline._import_video(context)

        assert result["status"] == StepStatus.COMPLETED.value
        assert result["video_guid"] == "test_video_123"
        assert "video_folder" in result
        assert "video_blob_name" in result

    def test_import_video_missing_path(self, pipeline):
        """Test video import with missing video path."""
        context = {}

        result = pipeline._import_video(context)

        assert result["status"] == StepStatus.ERROR.value
        assert "No video path provided" in result["error"]

    def test_import_video_resolution_too_low(self, pipeline, mock_storage):
        """Test video import with resolution too low."""
        # Mock video capture for resolution validation
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720
        }.get(prop, 0)
        mock_cap.__enter__ = Mock(return_value=mock_cap)
        mock_cap.__exit__ = Mock(return_value=None)
        mock_storage.get_video_capture.return_value = mock_cap

        context = {"raw_video_path": "raw_videos/low_res.mp4"}

        result = pipeline._import_video(context)

        assert result["status"] == StepStatus.ERROR.value
        assert "resolution" in result["error"].lower()

    def test_import_video_unopenable(self, pipeline, mock_storage):
        """Test video import with unopenable video."""
        # Mock video capture that fails to open
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cap.__enter__ = Mock(return_value=mock_cap)
        mock_cap.__exit__ = Mock(return_value=None)
        mock_storage.get_video_capture.return_value = mock_cap

        context = {"raw_video_path": "raw_videos/corrupt.mp4"}

        result = pipeline._import_video(context)

        assert result["status"] == StepStatus.ERROR.value
        assert "Could not open video" in result["error"]

    def test_is_crop_quality_sufficient_good_crop(self, pipeline):
        """Test crop quality check with a good quality crop."""
        # Create a crop with good contrast and size (100x60 pixels)
        # Use high contrast values (black and white stripes)
        crop = np.zeros((100, 60, 3), dtype=np.uint8)
        crop[:, :30] = [0, 0, 0]  # Black left half
        crop[:, 30:] = [255, 255, 255]  # White right half
        
        assert pipeline._is_crop_quality_sufficient(crop) is True

    def test_is_crop_quality_sufficient_too_small(self, pipeline):
        """Test crop quality check with crop that's too small."""
        crop = np.ones((40, 30, 3), dtype=np.uint8) * 128
        
        assert pipeline._is_crop_quality_sufficient(crop) is False

    def test_is_crop_quality_sufficient_low_contrast(self, pipeline):
        """Test crop quality check with low contrast crop."""
        # Create a crop with uniform color (low contrast)
        crop = np.full((100, 60, 3), 128, dtype=np.uint8)
        
        assert pipeline._is_crop_quality_sufficient(crop) is False

    @pytest.mark.asyncio
    async def test_execute_parallel_operations_async_empty_tasks(self, pipeline):
        """Test parallel operations with empty task list."""
        failed, successful = await pipeline._execute_parallel_operations_async(
            [], Mock(return_value=True), "test operation"
        )

        assert failed == []
        assert successful == 0

    @pytest.mark.asyncio
    async def test_execute_parallel_operations_async_success(self, pipeline):
        """Test parallel operations with successful tasks."""
        mock_operation = Mock(return_value=True)
        mock_operation.__name__ = "mock_operation"
        tasks = ["task1", "task2", "task3"]

        failed, successful = await pipeline._execute_parallel_operations_async(
            tasks, mock_operation, "test operation"
        )

        assert failed == []
        assert successful == 3
        assert mock_operation.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_parallel_operations_async_with_failures(self, pipeline):
        """Test parallel operations with some failures."""
        def mock_operation(task):
            return task != "task2"  # task2 fails
        
        tasks = ["task1", "task2", "task3"]

        failed, successful = await pipeline._execute_parallel_operations_async(
            tasks, mock_operation, "test operation"
        )

        assert len(failed) == 1
        assert failed[0] == "task2"
        assert successful == 2

    def test_workflow_instantiation(self):
        """Test that the UnverifiedTrackGenerationWorkflow can be instantiated."""
        from workflows.create_unverified_tracks import UnverifiedTrackGenerationWorkflow

        workflow = UnverifiedTrackGenerationWorkflow(
            tenant_id="test_tenant",
            verbose=False,
            custom_name="test_workflow"
        )

        assert workflow.tenant_id == "test_tenant"
        assert workflow.verbose is False
        assert workflow.custom_name == "test_workflow"
        assert workflow.detection_config is not None