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