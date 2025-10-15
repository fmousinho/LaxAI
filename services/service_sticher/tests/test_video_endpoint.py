import pytest
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