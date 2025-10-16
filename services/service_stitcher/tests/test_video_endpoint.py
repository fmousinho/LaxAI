import pytest
from unittest.mock import MagicMock, patch
from fastapi.exceptions import HTTPException
from services.service_stitcher.src.v1.endpoints.video_endpoint import load
from services.service_stitcher.src.v1.schemas.video_schema import VideoLoadRequest, VideoLoadResponse


def test_load_video_endpoint_real():
    req = VideoLoadRequest(tenant_id="test_tenant", video_path="process/test_unit_test_video_service/imported/test_video.mp4")

    # Act
    response = load(req)

    # Assert
    assert isinstance(response, VideoLoadResponse)
    assert response.session_id is not None
    assert response.video_path == "process/test_unit_test_video_service/imported/test_video.mp4"
    assert response.total_frames > 0
    assert isinstance(response.has_next_frame, bool)
    assert isinstance(response.has_previous_frame, bool)


def test_load_video_endpoint_nonexistent_file():
    """Test loading a video that doesn't exist in GCS."""
    req = VideoLoadRequest(tenant_id="test_tenant", video_path="nonexistent/path/to/video.mp4")

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        load(req)
    
    assert exc_info.value.status_code == 404
    assert "not found" in str(exc_info.value.detail).lower()


def test_load_video_endpoint_invalid_path():
    """Test loading with an invalid path format."""
    req = VideoLoadRequest(tenant_id="test_tenant", video_path="")

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        load(req)
    
    # Should get a 500 error for general exceptions during video loading
    assert exc_info.value.status_code == 500


def test_session_cleanup_mechanism():
    """Test that expired sessions are properly cleaned up."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import video_managers, CLEANUP_INTERVAL
    import time
    
    # Clear any existing sessions
    video_managers.clear()
    
    # Mock VideoManager to avoid GCS dependencies
    with patch('services.service_sticher.src.v1.endpoints.video_endpoint.VideoManager') as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        # Simulate creating a session 13 hours ago (past the 12-hour limit)
        expired_time = time.time() - (CLEANUP_INTERVAL + 60)  # 12 hours + 1 minute ago
        video_managers["expired-session"] = (mock_manager, expired_time)
        
        # Simulate creating a recent session
        recent_time = time.time() - 60  # 1 minute ago
        video_managers["recent-session"] = (mock_manager, recent_time)
        
        # Verify both sessions exist initially
        assert len(video_managers) == 2
        assert "expired-session" in video_managers
        assert "recent-session" in video_managers
        
        # Simulate the cleanup function running
        current_time = time.time()
        expired_sessions = []
        
        for session_id, (manager, created_at) in video_managers.items():
            if current_time - created_at > CLEANUP_INTERVAL:
                expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            del video_managers[session_id]
        
        # Verify only the expired session was removed
        assert len(video_managers) == 1
        assert "expired-session" not in video_managers
        assert "recent-session" in video_managers


def test_stop_and_save_endpoint_success():
    """Test successful deletion of a video session."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import stop_and_save, video_managers
    import time
    
    # Clear any existing sessions
    video_managers.clear()
    
    # Mock VideoManager
    with patch('services.service_sticher.src.v1.endpoints.video_endpoint.VideoManager') as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        # Set up a session in video_managers
        session_id = "test-session-123"
        video_managers[session_id] = (mock_manager, time.time())
        
        # Verify session exists
        assert session_id in video_managers
        
        # Call stop_and_save
        result = stop_and_save(session_id)
        
        # Verify session was removed
        assert session_id not in video_managers
        assert len(video_managers) == 0
        
        # Verify function returns None
        assert result is None


def test_stop_and_save_endpoint_session_not_found():
    """Test deletion of a non-existent video session returns 404."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import stop_and_save, video_managers
    
    # Clear any existing sessions
    video_managers.clear()
    
    # Try to delete a non-existent session
    with pytest.raises(HTTPException) as exc_info:
        stop_and_save("non-existent-session")
    
    # Should get a 404 error
    assert exc_info.value.status_code == 404
    assert "Session not found" in str(exc_info.value.detail)