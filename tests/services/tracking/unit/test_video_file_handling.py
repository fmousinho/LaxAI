"""
Basic video file tests for tracking service.

This module contains tests that validate video file handling
without running the full tracking pipeline.
"""

import pytest
import cv2
import os
from pathlib import Path


class TestVideoFileHandling:
    """Tests for basic video file operations."""

    @pytest.fixture
    def test_video_path(self):
        """Path to the test video file."""
        return Path(__file__).parent.parent / "test_data" / "test_video.mp4"

    def test_video_file_exists(self, test_video_path):
        """Test that the test video file exists."""
        assert test_video_path.exists(), f"Test video file not found at {test_video_path}"

    def test_video_file_can_be_opened(self, test_video_path):
        """Test that the video file can be opened with OpenCV."""
        cap = cv2.VideoCapture(str(test_video_path))

        try:
            assert cap.isOpened(), f"Failed to open video file: {test_video_path}"

            # Check basic video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            assert frame_count > 0, "Video has no frames"
            assert width > 0, "Video has invalid width"
            assert height > 0, "Video has invalid height"
            assert fps > 0, "Video has invalid FPS"

            print(f"Video properties: {frame_count} frames, {width}x{height}, {fps} FPS")

        finally:
            cap.release()

    def test_video_frames_can_be_read(self, test_video_path):
        """Test that frames can be read from the video file."""
        cap = cv2.VideoCapture(str(test_video_path))

        try:
            assert cap.isOpened(), f"Failed to open video file: {test_video_path}"

            # Try to read first few frames
            frames_read = 0
            max_frames_to_test = 5

            for i in range(max_frames_to_test):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_read += 1

                # Basic frame validation
                assert frame is not None, f"Frame {i} is None"
                assert frame.shape[2] == 3, f"Frame {i} doesn't have 3 color channels"
                assert frame.shape[0] > 0 and frame.shape[1] > 0, f"Frame {i} has invalid dimensions"

            assert frames_read > 0, "Could not read any frames from video"

            print(f"Successfully read {frames_read} frames from video")

        finally:
            cap.release()