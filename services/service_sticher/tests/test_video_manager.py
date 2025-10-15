import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from services.service_sticher.src.video_manager import VideoManager


class TestVideoManager:
    """Unit tests for VideoManager class focusing on next_frame and previous_frame methods."""

    @pytest.fixture
    def mock_video_manager(self):
        """Create a VideoManager with mocked GCSVideoCapture."""
        with patch('services.service_sticher.src.video_manager.get_storage') as mock_get_storage:
            mock_storage = MagicMock()
            mock_get_storage.return_value = mock_storage

            # Create VideoManager instance
            manager = VideoManager("test_tenant")

            # Mock the GCSVideoCapture
            mock_cap = MagicMock()
            manager.cap = mock_cap

            # Set up basic video properties
            manager.total_frames = 300  # 300 frames total
            manager.frame_skip_interval = 30  # Skip 30 frames
            manager.current_frame = None  # Start with no current frame

            yield manager, mock_cap

    def test_next_frame_no_video_loaded(self):
        """Test next_frame when no video is loaded."""
        with patch('services.service_sticher.src.video_manager.get_storage'):
            manager = VideoManager("test_tenant")
            manager.cap = None  # No video loaded

            with pytest.raises(ValueError, match="No video loaded for this session"):
                manager.next_frame()

    def test_next_frame_first_call(self, mock_video_manager):
        """Test next_frame on first call (starts at frame 0)."""
        manager, mock_cap = mock_video_manager

        # Mock successful frame read
        mock_frame = np.random.rand(480, 640, 3)  # RGB frame
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True

        result = manager.next_frame()

        # Should start at frame 0
        assert result["frame_id"] == 0
        assert result["has_next_frame"] is True  # 0 + 30 < 300
        assert result["has_previous_frame"] is False  # 0 > 0 is False
        assert result["frame_data"] is mock_frame

        # Should not call set on first frame
        mock_cap.set.assert_not_called()

    def test_next_frame_normal_advance(self, mock_video_manager):
        """Test next_frame advancing normally."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 0  # Simulate after first call

        # Mock successful frame read
        mock_frame = np.random.rand(480, 640, 3)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True

        result = manager.next_frame()

        # Should advance by frame_skip_interval (30)
        assert result["frame_id"] == 30
        assert result["has_next_frame"] is True  # 30 + 30 < 300
        assert result["has_previous_frame"] is True  # 30 > 0
        assert result["frame_data"] is mock_frame

        # Should seek to frame 30
        mock_cap.set.assert_called_once_with("CAP_PROP_POS_FRAMES", 30)

    def test_next_frame_end_of_video(self, mock_video_manager):
        """Test next_frame when reaching end of video."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 270  # 270 + 30 = 300, which >= 300

        with pytest.raises(ValueError, match="End of video reached"):
            manager.next_frame()

    def test_next_frame_seek_failure(self, mock_video_manager):
        """Test next_frame when frame seeking fails."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 0

        # Mock seek failure
        mock_cap.set.return_value = False

        with pytest.raises(ValueError, match="Failed to set frame position to 30"):
            manager.next_frame()

    def test_next_frame_read_failure(self, mock_video_manager):
        """Test next_frame when frame reading fails."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 0

        # Mock read failure
        mock_cap.read.return_value = (False, None)
        mock_cap.set.return_value = True

        with pytest.raises(ValueError, match="Failed to read frame at position 30"):
            manager.next_frame()

    def test_next_frame_boundary_last_frame(self, mock_video_manager):
        """Test next_frame at boundary where next advance would exceed video."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 240  # 240 + 30 = 270, which < 300

        mock_frame = np.random.rand(480, 640, 3)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True

        result = manager.next_frame()

        assert result["frame_id"] == 270
        assert result["has_next_frame"] is False  # 270 + 30 >= 300
        assert result["has_previous_frame"] is True

    def test_previous_frame_no_video_loaded(self):
        """Test previous_frame when no video is loaded."""
        with patch('services.service_sticher.src.video_manager.get_storage'):
            manager = VideoManager("test_tenant")
            manager.cap = None

            with pytest.raises(ValueError, match="No video loaded for this session"):
                manager.previous_frame()

    def test_previous_frame_first_call(self, mock_video_manager):
        """Test previous_frame on first call (stays at frame 0)."""
        manager, mock_cap = mock_video_manager

        mock_frame = np.random.rand(480, 640, 3)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True

        result = manager.previous_frame()

        # Should stay at frame 0
        assert result["frame_id"] == 0
        assert result["has_next_frame"] is True  # 0 + 30 < 300
        assert result["has_previous_frame"] is False  # 0 > 0 is False
        assert result["frame_data"] is mock_frame

        # Should not call set on first frame
        mock_cap.set.assert_not_called()

    def test_previous_frame_normal_rewind(self, mock_video_manager):
        """Test previous_frame rewinding normally."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 60  # Simulate being at frame 60

        mock_frame = np.random.rand(480, 640, 3)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True

        result = manager.previous_frame()

        # Should rewind by frame_skip_interval (30) to frame 30
        assert result["frame_id"] == 30
        assert result["has_next_frame"] is True  # 30 + 30 < 300
        assert result["has_previous_frame"] is True  # 30 > 0
        assert result["frame_data"] is mock_frame

        # Should seek to frame 30
        mock_cap.set.assert_called_once_with("CAP_PROP_POS_FRAMES", 30)

    def test_previous_frame_before_start(self, mock_video_manager):
        """Test previous_frame when going before video start (clamps to 0)."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 15  # 15 - 30 = -15, should clamp to 0

        mock_frame = np.random.rand(480, 640, 3)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True

        result = manager.previous_frame()

        # Should clamp to frame 0
        assert result["frame_id"] == 0
        assert result["has_next_frame"] is True
        assert result["has_previous_frame"] is False
        assert result["frame_data"] is mock_frame

        # Should seek to frame 0
        mock_cap.set.assert_called_once_with("CAP_PROP_POS_FRAMES", 0)

    def test_previous_frame_seek_failure(self, mock_video_manager):
        """Test previous_frame when frame seeking fails."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 60

        mock_cap.set.return_value = False

        with pytest.raises(ValueError, match="Failed to set frame position to 30"):
            manager.previous_frame()

    def test_previous_frame_read_failure(self, mock_video_manager):
        """Test previous_frame when frame reading fails."""
        manager, mock_cap = mock_video_manager
        manager.current_frame = 60

        mock_cap.read.return_value = (False, None)
        mock_cap.set.return_value = True

        with pytest.raises(ValueError, match="Failed to read frame at position 30"):
            manager.previous_frame()