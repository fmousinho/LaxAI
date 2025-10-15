import pytest
from unittest.mock import MagicMock, patch
from fastapi.exceptions import HTTPException
from services.service_sticher.src.v1.endpoints.video_endpoint import load
from services.service_sticher.src.v1.schemas.video_schema import VideoLoadRequest, VideoLoadResponse


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