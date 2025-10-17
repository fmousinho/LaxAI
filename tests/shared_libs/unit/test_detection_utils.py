"""
Unit tests for detection_utils.py

Tests the detection serialization/deserialization functions and helper utilities.
"""

import json
import pytest
import numpy as np
from typing import Dict, Any, List

from supervision import Detections
from shared_libs.common.detection_utils import (
    detections_to_json,
    json_to_detections,
    create_frame_response,
    update_player_mapping
)
from shared_libs.common.rendering_config import RenderingConfig, StylePreset


class TestDetectionUtils:
    """Unit tests for detection utility functions."""

    @pytest.fixture
    def sample_detections(self) -> Detections:
        """Create sample detections for testing."""
        # Create sample detection data
        xyxy = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120]
        ], dtype=np.float32)

        confidence = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        class_id = np.array([1, 1, 2], dtype=int)
        tracker_id = np.array([100, 101, 102], dtype=int)

        # Custom data
        data = {
            "player_id": np.array([1, 2, 3], dtype=int),
            "custom_field": ["value1", "value2", "value3"]
        }

        detections = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=data
        )

        # Add metadata
        detections.metadata = {"frame_id": 42, "video_id": "test_video"}

        return detections

    @pytest.fixture
    def sample_rendering_config(self) -> RenderingConfig:
        """Create sample rendering config for testing."""
        config = RenderingConfig()
        config.set_style_for_player(1, StylePreset.HIGHLIGHTED.value)
        config.set_style_for_player(2, StylePreset.DIMMED.value)
        config.set_color_for_player(1, (255, 0, 0))  # Red
        return config

    def test_detections_to_json_empty(self):
        """Test detections_to_json with empty detections."""
        # Test with empty list
        result = detections_to_json([])
        assert result == []

        # Test with empty Detections object
        empty_detections = Detections.empty()
        result = detections_to_json(empty_detections)
        assert result == []

    def test_detections_to_json_basic(self, sample_detections):
        """Test basic detections_to_json functionality."""
        result = detections_to_json(sample_detections)

        assert isinstance(result, list)
        assert len(result) == 3  # 3 detections

        # Check first detection
        first_det = result[0]
        assert "xyxy" in first_det
        assert "confidence" in first_det
        assert "class_id" in first_det
        assert "tracker_id" in first_det
        assert "data" in first_det
        assert "metadata" in first_det

        # Verify data types are JSON-serializable
        json_str = json.dumps(result)
        assert json_str  # Should not raise exception

    def test_detections_to_json_with_rendering_config(self, sample_detections, sample_rendering_config):
        """Test detections_to_json with include_rendering_config=True."""
        result = detections_to_json(
            sample_detections,
            rendering_config=sample_rendering_config,
            include_rendering_config=True
        )

        assert isinstance(result, dict)
        assert "detections" in result
        assert "rendering_config" in result

        # Check detections structure
        detections_data = result["detections"]
        assert "xyxy" in detections_data
        assert "confidence" in detections_data
        assert "class_id" in detections_data
        assert "tracker_id" in detections_data
        assert "data" in detections_data
        assert "metadata" in detections_data

        # Check rendering config
        config_data = result["rendering_config"]
        assert "player_styles" in config_data
        assert "custom_colors" in config_data

    def test_json_to_detections_basic(self, sample_detections):
        """Test basic json_to_detections functionality."""
        # Convert to JSON first
        json_data = detections_to_json(sample_detections)

        # Convert back
        reconstructed = json_to_detections(json_data)

        assert isinstance(reconstructed, Detections)
        assert len(reconstructed) == len(sample_detections)

        # Compare key attributes
        np.testing.assert_array_equal(reconstructed.xyxy, sample_detections.xyxy)
        np.testing.assert_array_equal(reconstructed.confidence, sample_detections.confidence)
        np.testing.assert_array_equal(reconstructed.class_id, sample_detections.class_id)
        np.testing.assert_array_equal(reconstructed.tracker_id, sample_detections.tracker_id)

        # Check custom data
        assert "player_id" in reconstructed.data
        assert "custom_field" in reconstructed.data
        np.testing.assert_array_equal(reconstructed.data["player_id"], sample_detections.data["player_id"])

    def test_json_to_detections_with_rendering_config(self, sample_detections, sample_rendering_config):
        """Test json_to_detections with rendering config."""
        # Convert to JSON with rendering config
        json_data = detections_to_json(
            sample_detections,
            rendering_config=sample_rendering_config,
            include_rendering_config=True
        )

        # Convert back with return_rendering_config=True
        reconstructed_detections, reconstructed_config = json_to_detections(
            json_data,
            return_rendering_config=True
        )

        assert isinstance(reconstructed_detections, Detections)
        assert isinstance(reconstructed_config, RenderingConfig)

        # Compare detections
        np.testing.assert_array_equal(reconstructed_detections.xyxy, sample_detections.xyxy)

        # Compare rendering config
        assert reconstructed_config.get_style_for_player(1) == StylePreset.HIGHLIGHTED.value
        assert reconstructed_config.get_style_for_player(2) == StylePreset.DIMMED.value
        assert reconstructed_config.get_color_for_player(1) == (255, 0, 0)

    def test_round_trip_serialization(self, sample_detections, sample_rendering_config):
        """Test complete round-trip: detections -> json -> detections."""
        # Forward conversion
        json_data = detections_to_json(
            sample_detections,
            rendering_config=sample_rendering_config,
            include_rendering_config=True
        )

        # Backward conversion
        result = json_to_detections(json_data, return_rendering_config=True)
        reconstructed_detections, reconstructed_config = result

        # Verify detections match
        assert len(reconstructed_detections) == len(sample_detections)
        np.testing.assert_array_equal(reconstructed_detections.xyxy, sample_detections.xyxy)
        np.testing.assert_array_equal(reconstructed_detections.confidence, sample_detections.confidence)
        np.testing.assert_array_equal(reconstructed_detections.class_id, sample_detections.class_id)
        np.testing.assert_array_equal(reconstructed_detections.tracker_id, sample_detections.tracker_id)

        # Verify custom data
        for key in sample_detections.data:
            if isinstance(sample_detections.data[key], np.ndarray):
                np.testing.assert_array_equal(
                    reconstructed_detections.data[key],
                    sample_detections.data[key]
                )
            else:
                assert reconstructed_detections.data[key] == sample_detections.data[key]

        # Verify rendering config
        assert reconstructed_config.get_style_for_player(1) == sample_rendering_config.get_style_for_player(1)
        assert reconstructed_config.get_color_for_player(1) == sample_rendering_config.get_color_for_player(1)

    def test_create_frame_response(self, sample_detections, sample_rendering_config):
        """Test create_frame_response helper function."""
        result = create_frame_response(
            frame_id=42,
            video_id="test_video",
            session_id="session_123",
            detections=sample_detections,
            rendering_config=sample_rendering_config,
            has_next=True,
            has_previous=False,
            total_frames=100
        )

        assert isinstance(result, dict)

        # Check metadata
        assert result["frame_id"] == 42
        assert result["video_id"] == "test_video"
        assert result["session_id"] == "session_123"
        assert result["has_next"] is True
        assert result["has_previous"] is False
        assert result["total_frames"] == 100

        # Check data structure
        assert "detections" in result
        assert "rendering_config" in result

        # Verify detections data is present
        detections_data = result["detections"]
        assert len(detections_data["xyxy"]) == 3  # 3 detections

    def test_update_player_mapping(self, sample_detections, sample_rendering_config):
        """Test update_player_mapping helper function."""
        # Original state
        original_player_ids = sample_detections.data["player_id"].copy()

        # Update mapping: tracker_id 101 (index 1) should get new player_id 99
        updated_detections, updated_config = update_player_mapping(
            detections=sample_detections,
            rendering_config=sample_rendering_config,
            tracker_id=101,
            new_player_id=99,
            style_preset=StylePreset.SUCCESS.value
        )

        # Check that detections were updated
        assert updated_detections.data["player_id"][1] == 99  # Index 1 corresponds to tracker_id 101
        assert updated_detections.data["player_id"][0] == original_player_ids[0]  # Other entries unchanged
        assert updated_detections.data["player_id"][2] == original_player_ids[2]

        # Check that rendering config was updated
        assert updated_config.get_style_for_player(99) == StylePreset.SUCCESS.value

    def test_backward_compatibility(self, sample_detections):
        """Test that old API still works (backward compatibility)."""
        # Old-style usage without rendering config
        json_data = detections_to_json(sample_detections)
        reconstructed = json_to_detections(json_data)

        assert isinstance(reconstructed, Detections)
        assert len(reconstructed) == len(sample_detections)

        # Should not return tuple when return_rendering_config=False (default)
        assert not isinstance(reconstructed, tuple)

    def test_empty_detections_edge_cases(self):
        """Test edge cases with empty detections."""
        empty_detections = Detections.empty()

        # Test with include_rendering_config=True
        result = detections_to_json(empty_detections, include_rendering_config=True)
        assert isinstance(result, dict)
        assert "detections" in result
        assert "rendering_config" in result
        assert len(result["detections"]["xyxy"]) == 0

        # Test json_to_detections with empty data
        reconstructed = json_to_detections([])
        assert isinstance(reconstructed, Detections)
        assert len(reconstructed) == 0

        # Test with return_rendering_config=True
        reconstructed_det, reconstructed_config = json_to_detections([], return_rendering_config=True)
        assert isinstance(reconstructed_det, Detections)
        assert isinstance(reconstructed_config, RenderingConfig)
        assert len(reconstructed_det) == 0

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid input to json_to_detections (non-dict/non-list)
        with pytest.raises((TypeError, AttributeError)):
            json_to_detections(123)  # type: ignore # Invalid type

        # Test invalid detections input to detections_to_json
        with pytest.raises(TypeError):
            detections_to_json(123)  # type: ignore # Invalid type