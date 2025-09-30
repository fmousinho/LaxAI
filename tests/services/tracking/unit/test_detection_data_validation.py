"""
Unit tests for detection data validation.

This module contains tests that verify the generated detection test data
can be properly loaded and converted to Detections objects.
"""

import json
from pathlib import Path

import pytest

from shared_libs.common.detection_utils import json_to_detections


class TestDetectionDataValidation:
    """Test suite for validating detection test data."""

    @pytest.fixture
    def detections_json_path(self):
        """Get the path to the test detections JSON file."""
        return Path(__file__).parent.parent / "test_data" / "detections.json"

    def test_detections_json_file_exists(self, detections_json_path):
        """Test that the detections.json file exists."""
        assert detections_json_path.exists(), f"Detections JSON file not found at {detections_json_path}"

    def test_detections_json_can_be_loaded(self, detections_json_path):
        """Test that the detections.json file can be loaded as valid JSON."""
        with open(detections_json_path, 'r') as f:
            data = json.load(f)

        assert "detections" in data, "JSON should contain a 'detections' key"
        assert isinstance(data["detections"], list), "'detections' should be a list"
        assert len(data["detections"]) > 0, "Should have at least one detection frame"

    def test_detections_json_can_be_converted_to_detections_objects(self, detections_json_path):
        """Test that the JSON data can be converted back to Detections objects."""
        # Load the JSON data
        with open(detections_json_path, 'r') as f:
            data = json.load(f)

        # Convert each frame back to Detections object
        detections_list = []
        for frame_data in data["detections"]:
            detections = json_to_detections([frame_data])
            detections_list.append(detections)

        # Verify we got valid detections
        assert len(detections_list) > 0, "Should have converted at least one frame"

        # Check that each detection has the expected properties
        for detections in detections_list:
            assert len(detections) > 0, "Each frame should have at least one detection"
            assert hasattr(detections, 'xyxy'), "Detections should have xyxy attribute"
            assert hasattr(detections, 'confidence'), "Detections should have confidence attribute"
            assert hasattr(detections, 'class_id'), "Detections should have class_id attribute"

            # Check data types and shapes
            assert detections.xyxy.ndim == 2, "xyxy should be 2D array"
            assert detections.xyxy.shape[1] == 4, "xyxy should have 4 coordinates per detection"
            assert len(detections.confidence) == len(detections), "Confidence should match number of detections"
            assert len(detections.class_id) == len(detections), "Class ID should match number of detections"

    def test_detections_have_valid_frame_indices(self, detections_json_path):
        """Test that detections include valid frame index data."""
        with open(detections_json_path, 'r') as f:
            data = json.load(f)

        for frame_data in data["detections"]:
            assert "data" in frame_data, "Frame should have data field"
            assert "frame_index" in frame_data["data"], "Frame should have frame_index in data"
            frame_indices = frame_data["data"]["frame_index"]
            assert isinstance(frame_indices, list), "frame_index should be a list"
            assert len(frame_indices) > 0, "frame_index should not be empty"

    def test_detections_have_valid_metadata(self, detections_json_path):
        """Test that detections include valid metadata."""
        with open(detections_json_path, 'r') as f:
            data = json.load(f)

        for frame_data in data["detections"]:
            assert "metadata" in frame_data, "Frame should have metadata"
            metadata = frame_data["metadata"]
            assert "frame_id" in metadata, "Metadata should include frame_id"
            assert "resolution" in metadata, "Metadata should include resolution"