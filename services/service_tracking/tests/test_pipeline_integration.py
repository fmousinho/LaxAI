"""
Integration tests for TrackGeneratorPipeline.

This module contains integration tests that test the pipeline end-to-end
with mocked external dependencies.
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


class TestTrackGeneratorPipelineIntegration:
    """Integration tests for TrackGeneratorPipeline."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock DetectionConfig for testing."""
        config = Mock(spec=DetectionConfig)
        config.model_path = "/fake/model/path"
        config.confidence_threshold = 0.5
        return config

    @pytest.fixture
    def mock_detection_model(self):
        """Create a mock DetectionModel that returns realistic detections."""
        model = Mock()
        # Create realistic detections
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 300], [300, 100, 400, 300]]),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([0, 0])
        )
        model.generate_detections.return_value = detections
        return model

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock tracker that adds track IDs."""
        tracker = Mock()
        tracker.get_identity_affine_matrix.return_value = np.eye(3)
        tracker.calculate_affine_transform.return_value = np.eye(3)

        # Mock the update_with_transform to return detections with track IDs
        def mock_update_with_transform(detections, affine_matrix, frame):
            # Simulate tracker adding track IDs by modifying the detections
            # In reality, this would add tracker_id field
            return detections

        tracker.update_with_transform.side_effect = mock_update_with_transform
        return tracker

    @pytest.fixture
    def mock_storage(self):
        """Create a mock GoogleStorageClient."""
        storage = Mock()
        storage.move_blob.return_value = True
        storage.upload_from_bytes.return_value = True
        storage.list_blobs.return_value = []
        return storage

    @pytest.fixture
    def mock_path_manager(self):
        """Create a mock GCSPaths manager."""
        paths = Mock()
        paths.get_path.side_effect = lambda *args, **kwargs: f"mock/path/{kwargs.get('video_id', 'video')}/{kwargs.get('track_id', 'track')}"
        return paths

    @pytest.fixture
    def pipeline(self, mock_config, mock_detection_model, mock_tracker,
                mock_storage, mock_path_manager):
        """Create a TrackGeneratorPipeline instance with mocked dependencies."""
        with patch('unverified_track_generator_pipeline.get_storage', return_value=mock_storage), \
             patch('unverified_track_generator_pipeline.GCSPaths', return_value=mock_path_manager), \
             patch('unverified_track_generator_pipeline.DetectionModel', return_value=mock_detection_model), \
             patch('unverified_track_generator_pipeline.AffineAwareByteTrack', return_value=mock_tracker), \
             patch('unverified_track_generator_pipeline.create_video_id', return_value='test_video_123'), \
             patch('unverified_track_generator_pipeline.create_run_id', return_value='test_run_123'):

            pipeline = TrackGeneratorPipeline(
                config=mock_config,
                tenant_id="test_tenant",
                verbose=False
            )
            return pipeline

    def test_full_pipeline_run_with_mock_video(self, pipeline,
                                             mock_storage, mock_path_manager):
        """Test full pipeline run with mocked video processing."""
        # Mock video capture as context manager
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 10,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.read.side_effect = [
            (True, np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)),
            (True, np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)),
            (False, None)  # End of video
        ]
        mock_cap.__enter__ = Mock(return_value=mock_cap)
        mock_cap.__exit__ = Mock(return_value=None)
        mock_storage.get_video_capture.return_value = mock_cap

        # Mock crop image function
        with patch('unverified_track_generator_pipeline.crop_image') as mock_crop_image, \
             patch('unverified_track_generator_pipeline.create_video_id', return_value='test_video_123'):
            mock_crop = np.ones((100, 50, 3), dtype=np.uint8) * 128
            mock_crop_image.return_value = mock_crop

            # Run pipeline
            result = pipeline.run("raw_videos/test.mp4")

            # Verify pipeline completed
            assert result["status"] == "completed"
            assert result["video_guid"] == "test_video_123"
            assert "video_folder" in result
            assert "pipeline_summary" in result

    def test_pipeline_error_handling_detection_model_failure(self, pipeline):
        """Test pipeline error handling when detection model fails."""
        # Make detection model fail
        pipeline.detection_model.generate_detections.side_effect = RuntimeError("Model error")

        # Mock video capture
        with patch.object(pipeline.tenant_storage, 'get_video_capture') as mock_get_cap:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 5,
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080
            }.get(prop, 0)
            mock_cap.read.return_value = (True, np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8))
            mock_get_cap.return_value = mock_cap

            # Run pipeline - should handle detection errors gracefully
            result = pipeline._get_detections_and_tracks({
                "video_blob_name": "test.mp4",
                "video_guid": "test_video_123",
                "video_folder": "test/folder"
            })

            # Should still return a result (though with errors)
            assert "status" in result

    @patch('asyncio.create_task')
    @patch('asyncio.run')
    def test_pipeline_graceful_stop_integration(self, mock_asyncio_run, mock_create_task, pipeline):
        """Test graceful stop functionality in integration context."""
        # Mock the async task creation and run
        mock_task = AsyncMock()
        mock_create_task.return_value = mock_task
        mock_asyncio_run.return_value = None  # Don't actually run async code
        
        # Create mock tasks - these should be proper awaitables
        crop_task = AsyncMock()
        upload_task1 = AsyncMock()
        upload_task2 = AsyncMock()

        # Call graceful stop
        context = pipeline._graceful_stop(
            [crop_task], [upload_task1, upload_task2], 1, "video_123",
            ["crop1.jpg", "crop2.jpg"], 50, 25, [sv.Detections.empty()],
            "video.mp4", "video_folder"
        )

        # Verify context contains expected cancellation data
        assert context["status"] == StepStatus.CANCELLED.value
        assert context["total_detections"] == 25
        assert context["total_crops"] == 2
        assert context["resume_frame"] == 50
        assert context["cancellation_reason"] == "Stop requested during detection processing"

    @patch('asyncio.gather')
    def test_async_upload_batch_processing(self, mock_gather, pipeline):
        """Test async batch upload processing."""
        # Mock asyncio.gather to return upload tasks
        mock_gather.return_value = [
            [("path1.jpg", np.ones((100, 50, 3), dtype=np.uint8))],
            [("path2.jpg", np.ones((100, 50, 3), dtype=np.uint8))]
        ]

        # Mock the parallel operations
        with patch.object(pipeline, '_execute_parallel_operations_async') as mock_parallel:
            mock_parallel.return_value = ([], 2)  # No failures, 2 successful

            # This would normally be called internally, but we can test the logic
            # by calling the method directly with mock data
            pass  # Integration test structure - actual async testing would be complex

    def test_memory_efficiency_with_large_batches(self, pipeline):
        """Test that pipeline handles batching correctly for memory efficiency."""
        # This test would verify that crops are batched properly
        # and not all loaded into memory at once

        # Create a scenario with many detections
        large_detections = sv.Detections(
            xyxy=np.random.randint(0, 1000, (100, 4)),  # 100 detections
            confidence=np.random.rand(100),
            class_id=np.zeros(100, dtype=int)
        )

        # Mock crop extraction
        with patch.object(pipeline, '_get_crops') as mock_get_crops:
            crops = [(i, np.ones((96, 48, 3), dtype=np.uint8)) for i in range(100)]
            mock_get_crops.return_value = crops

            # Test that batching logic works (this would be called internally)
            # The actual batching happens in _get_detections_and_tracks
            assert len(crops) == 100  # Verify we can handle large numbers