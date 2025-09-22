"""
Tests for the LaxAI API Service.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "LaxAI API Service" in data["message"]


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "laxai-api-service"


def test_list_training_jobs(client):
    """Test listing training jobs."""
    response = client.get("/train/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "jobs" in data


def test_list_tracking_jobs(client):
    """Test listing tracking jobs."""
    response = client.get("/track/")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "count" in data


def test_create_tracking_job_missing_tenant_id(client):
    """Test creating a tracking job with missing tenant_id."""
    request_data = {
        "custom_name": "test_tracking_job"
        # Missing tenant_id
    }
    response = client.post("/track/", json=request_data)
    assert response.status_code == 422  # Validation error


def test_create_tracking_job_valid(client, mocker):
    """Test creating a tracking job with valid data."""
    # Mock the PubSub publisher
    mock_publisher = mocker.patch("src.v1.endpoints.track.get_publisher")
    mock_instance = mocker.MagicMock()
    mock_instance.publish_tracking_request.return_value = "test-task-id-123"
    mock_publisher.return_value = mock_instance

    request_data = {
        "custom_name": "test_tracking_job",
        "tenant_id": "test-tenant",
        "tracking_params": {
            "verbose": True,
            "save_intermediate": False,
            "video_limit": 10
        }
    }

    response = client.post("/track/", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id-123"
    assert data["status"] == "queued"
    assert "Tracking job queued successfully" in data["message"]


def test_get_tracking_job_status_not_found(client, mocker):
    """Test getting status for a non-existent tracking job."""
    # Mock the status manager
    mock_manager = mocker.patch("src.v1.endpoints.track.get_status_manager")
    mock_instance = mocker.MagicMock()
    mock_instance.get_tracking_job_status.return_value = {
        "task_id": "non-existent-id",
        "status": "not_found",
        "error": "No tracking job found with this task_id."
    }
    mock_manager.return_value = mock_instance

    response = client.get("/track/non-existent-id")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "non-existent-id"
    assert data["status"] == "not_found"


def test_cancel_tracking_job(client, mocker):
    """Test cancelling a tracking job."""
    # Mock the status manager and its methods
    mock_manager = mocker.patch("src.v1.endpoints.track.get_status_manager")
    mock_instance = mocker.MagicMock()
    mock_instance.get_tracking_job_status.return_value = {
        "task_id": "test-task-id",
        "status": "running"
    }
    mock_instance._runs_collection = mocker.MagicMock()
    mock_manager.return_value = mock_instance

    response = client.delete("/track/test-task-id")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id"
    assert data["status"] == "cancelled"