"""
Independent test suite for TrackGeneratorPipeline functionality.

This test suite can be deployed and run independently from the main LaxAI test suite,
containing all necessary mocks and fixtures for testing the TrackGeneratorPipeline
in isolation.
"""

import json
import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Mock external dependencies
cv2_mock = Mock()
cv2_mock.__spec__ = Mock()  # Add __spec__ for transformers compatibility
sys.modules['cv2'] = cv2_mock
# sys.modules['supervision'] = Mock()  # Keep supervision real for imports
# sys.modules['torch'] = Mock()  # Keep torch real for torchvision compatibility
# sys.modules['torchvision'] = Mock()  # Keep torchvision real for transforms
# sys.modules['PIL'] = Mock()  # Keep PIL real for supervision compatibility
# sys.modules['PIL.Image'] = Mock()  # Keep PIL real for supervision compatibility

# Mock specific modules that might not be available
mock_modules = [
    'config.all_config',
    'config.training_config',
    'config.transforms',  # Mock transforms to avoid ColorJitter issues
    'modules.clustering_processor',
    'modules.crop_extractor_processor',
    'modules.emb_processor',
    'models.detection_model',
    'models.affine_tracker',
    'common.google_storage',
    'common.pipeline_step',
    'common.gcs_paths'
]

for module in mock_modules:
    sys.modules[module] = Mock()

# Now import the pipeline
from unverified_track_generator_pipeline import (CHECKPOINT_FRAME_INTERVAL,
                                                 TrackGeneratorPipeline)


class TestTrackGeneratorPipeline:
    """Test suite for TrackGeneratorPipeline functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock()
        config.frames_per_video = 100
        config.num_workers = 2
        return config

    @pytest.fixture
    def mock_storage_client(self):
        """Create a mock storage client."""
        storage = Mock()
        storage.bucket_name = "test-bucket"
        storage.upload_from_string = Mock(return_value=True)
        storage.download_as_string = Mock(return_value='{"test": "data"}')
        storage.blob_exists = Mock(return_value=True)
        storage.list_blobs = Mock(return_value=[])
        storage.delete_blob = Mock(return_value=True)

        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened = Mock(return_value=True)
        mock_cap.get = Mock(side_effect=lambda prop: {
            7: 1000,  # CAP_PROP_FRAME_COUNT
            1: 640,   # CAP_PROP_POS_FRAMES (current frame)
        }.get(prop, 0))
        mock_cap.read = Mock(return_value=(True, Mock()))  # (ret, frame)
        mock_cap.set = Mock(return_value=True)
        storage.get_video_capture = Mock(return_value=mock_cap)

        return storage

    @pytest.fixture
    def mock_detection_model(self):
        """Create a mock detection model."""
        model = Mock()
        model.generate_detections = Mock(return_value=Mock())  # Mock Detections object
        return model

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock tracker."""
        tracker = Mock()
        tracker.get_identity_affine_matrix = Mock(return_value=Mock())
        tracker.calculate_affine_transform = Mock(return_value=Mock())
        tracker.update_with_transform = Mock(return_value=Mock())
        return tracker

    @pytest.fixture
    def pipeline(self, mock_config, mock_storage_client, mock_detection_model, mock_tracker):
        """Create a TrackGeneratorPipeline instance with mocked dependencies."""
        # Mock the essential dependencies
        with patch('common.google_storage.get_storage', return_value=mock_storage_client), \
             patch('common.detection.DetectionModel', return_value=mock_detection_model), \
             patch('common.tracker.AffineAwareByteTrack', return_value=mock_tracker), \
             patch('common.google_storage.GCSPaths'), \
             patch('config.all_config.training_config', mock_config), \
             patch('src.track.unverified_track_generator_pipeline.DetectionModel', return_value=mock_detection_model), \
             patch('src.track.unverified_track_generator_pipeline.AffineAwareByteTrack', return_value=mock_tracker):

            # Import and create the pipeline
            from unverified_track_generator_pipeline import \
                TrackGeneratorPipeline
            
            pipeline = TrackGeneratorPipeline(mock_config, "test-tenant")
            yield pipeline

    def test_pipeline_initialization(self, pipeline):
        """Test that the pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.tenant_id == "test-tenant"
        assert hasattr(pipeline, 'run')
        assert hasattr(pipeline, 'stop')
        assert hasattr(pipeline, 'is_stop_requested')

    def test_constants_defined(self):
        """Test that required constants are properly defined."""
        assert CHECKPOINT_FRAME_INTERVAL > 0
        assert CHECKPOINT_FRAME_INTERVAL == 100

    def test_run_method_signature(self, pipeline):
        """Test that the run method has the correct signature."""
        import inspect
        sig = inspect.signature(pipeline.run)
        params = list(sig.parameters.keys())

        assert 'video_path' in params
        assert 'resume_from_checkpoint' in params

        # Check default value
        resume_param = sig.parameters['resume_from_checkpoint']
        assert resume_param.default is True

    def test_stop_functionality(self, pipeline):
        """Test the stop functionality."""
        assert not pipeline.is_stop_requested()

        pipeline.stop()
        assert pipeline.is_stop_requested()

    @patch('track.unverified_track_generator_pipeline.logger')
    def test_run_with_invalid_path(self, mock_logger, pipeline):
        """Test run method with invalid video path."""
        result = pipeline.run("")

        assert result["status"] == "error"
        assert "No video path provided" in result["error"]

    @patch('track.unverified_track_generator_pipeline.logger')
    def test_run_with_missing_dependencies(self, mock_logger, pipeline):
        """Test run method when dependencies are missing."""
        # Mock missing detection model
        pipeline.detection_model = None

        result = pipeline.run("test_video.mp4")

        assert result["status"] == "error"
        assert "Detection model not initialized" in result["error"]

    def test_checkpoint_constants(self):
        """Test that checkpoint-related constants are properly configured."""
        assert CHECKPOINT_FRAME_INTERVAL > 0
        assert isinstance(CHECKPOINT_FRAME_INTERVAL, int)

    def test_pipeline_inheritance(self, pipeline):
        """Test that pipeline properly inherits from base Pipeline class."""
        # Check that it has base pipeline methods
        assert hasattr(pipeline, 'save_checkpoint')
        assert hasattr(pipeline, 'current_completed_steps')

    @patch('track.unverified_track_generator_pipeline.logger')
    def test_video_capture_handling(self, mock_logger, pipeline, mock_storage_client):
        """Test that video capture is handled properly."""
        # Mock video capture to return False for isOpened
        mock_cap = Mock()
        mock_cap.isOpened = Mock(return_value=False)
        mock_storage_client.get_video_capture = Mock(return_value=mock_cap)

        result = pipeline.run("test_video.mp4")

        assert result["status"] == "error"
        assert "Could not open video" in result["error"]

    def test_context_initialization(self, pipeline):
        """Test that context is properly initialized."""
        context = {"raw_video_path": "test.mp4"}
        result = pipeline.run("test.mp4")

        # Should have basic result structure
        assert "status" in result
        assert "run_guid" in result
        assert "run_folder" in result
        assert "video_path" in result

    def test_resume_frame_in_results(self, pipeline):
        """Test that resume_frame is included in results when applicable."""
        result = pipeline.run("test.mp4")

        # resume_frame should be None for fresh runs
        assert result.get("resume_frame") is None

    def test_pipeline_summary_structure(self, pipeline):
        """Test that pipeline summary has expected structure."""
        result = pipeline.run("test.mp4")

        assert "pipeline_summary" in result
        summary = result["pipeline_summary"]
        assert isinstance(summary, dict)


class TestPipelineConstants:
    """Test pipeline constants and configuration."""

    def test_checkpoint_frame_interval(self):
        """Test CHECKPOINT_FRAME_INTERVAL constant."""
        assert CHECKPOINT_FRAME_INTERVAL == 100
        assert CHECKPOINT_FRAME_INTERVAL > 0

    def test_constants_are_integers(self):
        """Test that constants are integers."""
        assert isinstance(CHECKPOINT_FRAME_INTERVAL, int)
