import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import supervision as sv
from services.service_stitcher.src.video_manager import VideoManager


@pytest.mark.integration
class TestVideoManager:
    """Integration tests for VideoManager class using real GCS video."""

    @pytest.fixture
    def mock_video_manager(self):
        """Create a VideoManager with real GCS video loading."""
        manager = VideoManager("test_tenant")
        # Load the real test video from GCS
        manager.load_video("test_tenant/process/test_unit_test_video_service/imported/test_video.mp4")

        yield manager

    def test_next_frame_no_video_loaded(self):
        """Test next_frame when no video is loaded."""
        with patch('services.service_stitcher.src.video_manager.get_storage'):
            manager = VideoManager("test_tenant")
            manager.cap = None  # No video loaded

            with pytest.raises(ValueError, match="No video loaded for this session"):
                manager.next_frame()

    def test_next_frame_first_call(self, mock_video_manager):
        """Test next_frame on first call (starts at frame 0) - returns metadata only."""
        manager = mock_video_manager

        result = manager.next_frame()

        # Should start at frame 0
        assert result["frame_id"] == 0
        assert result["has_next_frame"] is True  # 0 + 30 < total_frames
        assert result["has_previous_frame"] is False  # 0 > 0 is False
        assert "detections" in result
        assert "player_mappings" in result
        assert "video_id" in result
        assert "session_id" in result
        # Should NOT include frame_data (client-side annotation)
        assert "frame_data" not in result

    def test_next_frame_normal_advance(self, mock_video_manager):
        """Test next_frame advancing normally - returns metadata only."""
        manager = mock_video_manager
        manager.current_frame_id = 0  # Simulate after first call

        result = manager.next_frame()

        # Should advance by frame_skip_interval (30)
        assert result["frame_id"] == 30
        assert result["has_next_frame"] is True  # 30 + 30 < total_frames
        assert result["has_previous_frame"] is True  # 30 > 0
        assert "detections" in result
        assert "frame_data" not in result  # Metadata only

    def test_next_frame_end_of_video(self, mock_video_manager):
        """Test next_frame when reaching end of video."""
        manager = mock_video_manager
        manager.current_frame_id = 270  # 270 + 30 = 300, which >= 300

        with pytest.raises(ValueError, match="End of video reached"):
            manager.next_frame()

    def test_next_frame_boundary_last_frame(self, mock_video_manager):
        """Test next_frame at boundary where next advance would exceed video."""
        manager = mock_video_manager
        manager.current_frame_id = 240  # 240 + 30 = 270, which < total_frames

        result = manager.next_frame()

        assert result["frame_id"] == 270
        assert result["has_next_frame"] is False  # 270 + 30 >= total_frames
        assert result["has_previous_frame"] is True
        assert "detections" in result
        assert "frame_data" not in result

    def test_previous_frame_no_video_loaded(self):
        """Test previous_frame when no video is loaded."""
        with patch('services.service_stitcher.src.video_manager.get_storage'):
            manager = VideoManager("test_tenant")
            manager.cap = None

            with pytest.raises(ValueError, match="No video loaded for this session"):
                manager.previous_frame()

    def test_previous_frame_first_call(self, mock_video_manager):
        """Test previous_frame on first call (stays at frame 0) - returns metadata only."""
        manager = mock_video_manager

        result = manager.previous_frame()

        # Should stay at frame 0
        assert result["frame_id"] == 0
        assert result["has_next_frame"] is True  # 0 + 30 < total_frames
        assert result["has_previous_frame"] is False  # 0 > 0 is False
        assert "detections" in result
        assert "frame_data" not in result  # Metadata only

    def test_previous_frame_normal_rewind(self, mock_video_manager):
        """Test previous_frame rewinding normally - returns metadata only."""
        manager = mock_video_manager
        manager.current_frame_id = 60  # Simulate being at frame 60

        result = manager.previous_frame()

        # Should rewind by frame_skip_interval (30) to frame 30
        assert result["frame_id"] == 30
        assert result["has_next_frame"] is True  # 30 + 30 < total_frames
        assert result["has_previous_frame"] is True  # 30 > 0
        assert "detections" in result
        assert "frame_data" not in result

    def test_previous_frame_before_start(self, mock_video_manager):
        """Test previous_frame when going before video start (clamps to 0)."""
        manager = mock_video_manager
        manager.current_frame_id = 15  # 15 - 30 = -15, should clamp to 0

        result = manager.previous_frame()

        # Should clamp to frame 0
        assert result["frame_id"] == 0
        assert result["has_next_frame"] is True
        assert result["has_previous_frame"] is False
        assert "detections" in result
        assert "frame_data" not in result
    
    def test_get_frame_metadata(self, mock_video_manager):
        """Test get_frame_metadata returns proper metadata structure."""
        manager = mock_video_manager
        
        result = manager.get_frame_metadata(0)
        
        # Verify metadata structure
        assert result["frame_id"] == 0
        assert "video_id" in result
        assert "session_id" in result
        assert "detections" in result
        assert "player_mappings" in result
        assert "has_next_frame" in result
        assert "has_previous_frame" in result
        assert "total_frames" in result
        assert "frame_data" not in result  # Should NOT include frame data
    
    def test_get_frame_metadata_invalid_frame(self, mock_video_manager):
        """Test get_frame_metadata with invalid frame_id."""
        manager = mock_video_manager
        
        # Test negative frame_id
        with pytest.raises(ValueError, match="Invalid frame_id"):
            manager.get_frame_metadata(-1)
        
        # Test frame_id beyond total_frames
        with pytest.raises(ValueError, match="Invalid frame_id"):
            manager.get_frame_metadata(999999)
    
    def test_get_raw_frame_image_png(self, mock_video_manager):
        """Test get_raw_frame_image returns PNG bytes."""
        manager = mock_video_manager
        
        image_bytes = manager.get_raw_frame_image(0, format="png")
        
        # Verify it's bytes
        assert isinstance(image_bytes, bytes)
        # PNG signature bytes
        assert image_bytes.startswith(b'\x89PNG')
    
    def test_get_raw_frame_image_jpeg(self, mock_video_manager):
        """Test get_raw_frame_image returns JPEG bytes."""
        manager = mock_video_manager
        
        image_bytes = manager.get_raw_frame_image(0, format="jpeg", quality=85)
        
        # Verify it's bytes
        assert isinstance(image_bytes, bytes)
        # JPEG signature bytes
        assert image_bytes.startswith(b'\xff\xd8\xff')
    
    def test_get_raw_frame_image_invalid_format(self, mock_video_manager):
        """Test get_raw_frame_image with unsupported format."""
        manager = mock_video_manager
        
        with pytest.raises(ValueError, match="Unsupported image format"):
            manager.get_raw_frame_image(0, format="bmp")
    
    def test_get_raw_frame_image_invalid_frame(self, mock_video_manager):
        """Test get_raw_frame_image with invalid frame_id."""
        manager = mock_video_manager
        
        with pytest.raises(ValueError, match="Invalid frame_id"):
            manager.get_raw_frame_image(-1)
        
        with pytest.raises(ValueError, match="Invalid frame_id"):
            manager.get_raw_frame_image(999999)