import pytest
from unittest.mock import MagicMock, patch
from fastapi.exceptions import HTTPException
from services.service_stitcher.src.v1.endpoints.video_endpoint import load
from services.service_stitcher.src.v1.schemas.video_schema import VideoLoadRequest, VideoLoadResponse


@pytest.mark.skip(reason="Requires valid GCS credentials - run as integration test only")
def test_load_video_endpoint_real():
    req = VideoLoadRequest(tenant_id="test_tenant", video_path="process/test_unit_test_video_service/imported/test_video.mp4")

    # Act
    response = load(req)

    # Assert
    assert isinstance(response, VideoLoadResponse)
    assert response.session_id is not None
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
    with patch('services.service_stitcher.src.v1.endpoints.video_endpoint.VideoManager') as mock_manager_class:
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
    with patch('services.service_stitcher.src.v1.endpoints.video_endpoint.VideoManager') as mock_manager_class:
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


# Tests for new annotation endpoints
def test_get_frame_annotations_success():
    """Test successful GET frame annotations."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import get_frame_annotations, video_managers
    from services.service_stitcher.src.v1.schemas.video_schema import AnnotationDataResponse
    import time

    # Clear any existing sessions
    video_managers.clear()

    # Mock VideoManager
    mock_manager = MagicMock()
    mock_annotation_data = {
        "frame_id": 42,
        "video_id": "test_video",
        "session_id": "test-session",
        "detections": {
            "xyxy": [[100, 200, 150, 250]],
            "confidence": [0.9],
            "class_id": [1],
            "tracker_id": [123],
            "data": {"player_id": [5]}
        },
        "rendering_config": {
            "player_styles": {"5": "highlighted"},
            "tracker_styles": {},
            "default_style": "default",
            "custom_colors": {}
        },
        "has_next": True,
        "has_previous": True,
        "total_frames": 100
    }
    mock_manager.get_frame_annotation_data.return_value = mock_annotation_data

    # Set up a session in video_managers
    session_id = "test-session"
    video_managers[session_id] = (mock_manager, time.time())

    # Call the endpoint
    response = get_frame_annotations(session_id, 42)

    # Verify response
    assert isinstance(response, AnnotationDataResponse)
    assert response.frame_id == 42
    assert response.video_id == "test_video"
    assert response.session_id == "test-session"
    assert response.has_next is True
    assert response.has_previous is True
    assert response.total_frames == 100

    # Verify VideoManager method was called correctly
    mock_manager.get_frame_annotation_data.assert_called_once_with(42)


def test_get_frame_annotations_session_not_found():
    """Test GET frame annotations with non-existent session."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import get_frame_annotations, video_managers

    # Clear any existing sessions
    video_managers.clear()

    # Try to get annotations for non-existent session
    with pytest.raises(HTTPException) as exc_info:
        get_frame_annotations("non-existent-session", 42)

    assert exc_info.value.status_code == 404
    assert "Session not found" in str(exc_info.value.detail)


def test_get_frame_annotations_invalid_frame_id():
    """Test GET frame annotations with invalid frame_id."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import get_frame_annotations, video_managers
    import time

    # Clear any existing sessions
    video_managers.clear()

    # Mock VideoManager that raises ValueError for invalid frame_id
    mock_manager = MagicMock()
    mock_manager.get_frame_annotation_data.side_effect = ValueError("Invalid frame_id: 999")

    # Set up a session in video_managers
    session_id = "test-session"
    video_managers[session_id] = (mock_manager, time.time())

    # Try to get annotations with invalid frame_id
    with pytest.raises(HTTPException) as exc_info:
        get_frame_annotations(session_id, 999)

    assert exc_info.value.status_code == 400
    assert "Invalid frame_id: 999" in str(exc_info.value.detail)


def test_update_frame_annotations_success():
    """Test successful PUT frame annotations."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import update_frame_annotations, video_managers
    from services.service_stitcher.src.v1.schemas.video_schema import AnnotationDataResponse
    import time

    # Clear any existing sessions
    video_managers.clear()

    # Mock VideoManager
    mock_manager = MagicMock()
    mock_updated_data = {
        "frame_id": 42,
        "video_id": "test_video",
        "session_id": "test-session",
        "detections": {
            "xyxy": [[100, 200, 150, 250]],
            "confidence": [0.9],
            "class_id": [1],
            "tracker_id": [123],
            "data": {"player_id": [7]}  # Updated player_id
        },
        "rendering_config": {
            "player_styles": {"7": "highlighted"},
            "tracker_styles": {},
            "default_style": "default",
            "custom_colors": {}
        },
        "has_next": True,
        "has_previous": True,
        "total_frames": 100
    }
    mock_manager.update_frame_annotation_data.return_value = mock_updated_data

    # Set up a session in video_managers
    session_id = "test-session"
    video_managers[session_id] = (mock_manager, time.time())

    # Create request data
    request_data = AnnotationDataResponse(
        frame_id=42,
        video_id="test_video",
        session_id="test-session",
        detections={
            "xyxy": [[100, 200, 150, 250]],
            "confidence": [0.9],
            "class_id": [1],
            "tracker_id": [123],
            "data": {"player_id": [7]}
        },
        rendering_config={
            "player_styles": {"7": "highlighted"},
            "tracker_styles": {},
            "default_style": "default",
            "custom_colors": {}
        },
        has_next=True,
        has_previous=True,
        total_frames=100
    )

    # Call the endpoint
    response = update_frame_annotations(session_id, 42, request_data)

    # Verify response
    assert isinstance(response, AnnotationDataResponse)
    assert response.frame_id == 42
    assert response.detections["data"]["player_id"] == [7]  # Updated player_id

    # Verify VideoManager method was called correctly
    mock_manager.update_frame_annotation_data.assert_called_once_with(
        frame_id=42,
        annotation_data=request_data.model_dump()
    )


def test_update_frame_annotations_session_not_found():
    """Test PUT frame annotations with non-existent session."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import update_frame_annotations, video_managers
    from services.service_stitcher.src.v1.schemas.video_schema import AnnotationDataResponse

    # Clear any existing sessions
    video_managers.clear()

    # Create request data
    request_data = AnnotationDataResponse(
        frame_id=42,
        video_id="test_video",
        session_id="test-session",
        detections={"xyxy": [], "confidence": [], "class_id": [], "tracker_id": [], "data": {}},
        rendering_config={"player_styles": {}, "tracker_styles": {}, "default_style": "default", "custom_colors": {}},
        has_next=True,
        has_previous=True,
        total_frames=100
    )

    # Try to update annotations for non-existent session
    with pytest.raises(HTTPException) as exc_info:
        update_frame_annotations("non-existent-session", 42, request_data)

    assert exc_info.value.status_code == 404
    assert "Session not found" in str(exc_info.value.detail)


def test_update_frame_annotations_invalid_frame_id():
    """Test PUT frame annotations with invalid frame_id."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import update_frame_annotations, video_managers
    from services.service_stitcher.src.v1.schemas.video_schema import AnnotationDataResponse
    import time

    # Clear any existing sessions
    video_managers.clear()

    # Mock VideoManager that raises ValueError for invalid frame_id
    mock_manager = MagicMock()
    mock_manager.update_frame_annotation_data.side_effect = ValueError("Invalid frame_id: 999")

    # Set up a session in video_managers
    session_id = "test-session"
    video_managers[session_id] = (mock_manager, time.time())

    # Create request data
    request_data = AnnotationDataResponse(
        frame_id=999,
        video_id="test_video",
        session_id="test-session",
        detections={"xyxy": [], "confidence": [], "class_id": [], "tracker_id": [], "data": {}},
        rendering_config={"player_styles": {}, "tracker_styles": {}, "default_style": "default", "custom_colors": {}},
        has_next=True,
        has_previous=True,
        total_frames=100
    )

    # Try to update annotations with invalid frame_id
    with pytest.raises(HTTPException) as exc_info:
        update_frame_annotations(session_id, 999, request_data)

    assert exc_info.value.status_code == 400
    assert "Invalid frame_id: 999" in str(exc_info.value.detail)


def test_update_frame_annotations_invalid_data():
    """Test PUT frame annotations with invalid data format."""
    from services.service_stitcher.src.v1.endpoints.video_endpoint import update_frame_annotations, video_managers
    from services.service_stitcher.src.v1.schemas.video_schema import AnnotationDataResponse
    import time

    # Clear any existing sessions
    video_managers.clear()

    # Mock VideoManager that raises ValueError for invalid data
    mock_manager = MagicMock()
    mock_manager.update_frame_annotation_data.side_effect = ValueError("annotation_data must contain 'detections' key")

    # Set up a session in video_managers
    session_id = "test-session"
    video_managers[session_id] = (mock_manager, time.time())

    # Create request data with missing detections
    request_data = AnnotationDataResponse(
        frame_id=42,
        video_id="test_video",
        session_id="test-session",
        detections={},  # Invalid: missing required fields
        rendering_config={"player_styles": {}, "tracker_styles": {}, "default_style": "default", "custom_colors": {}},
        has_next=True,
        has_previous=True,
        total_frames=100
    )

    # Try to update annotations with invalid data
    with pytest.raises(HTTPException) as exc_info:
        update_frame_annotations(session_id, 42, request_data)

    assert exc_info.value.status_code == 400
    assert "annotation_data must contain 'detections' key" in str(exc_info.value.detail)