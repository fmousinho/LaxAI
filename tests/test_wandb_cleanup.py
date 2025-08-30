import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from train.wandb_logger import wandb_logger


class MockVersion:
    def __init__(self, name: str, version: int, created_at: datetime, aliases=None, metadata=None, delete_raises=False):
        self.name = name
        self.version = version
        self.created_at = created_at
        self.aliases = aliases or []
        self.metadata = metadata or {}
        self._deleted = False
        self._delete_raises = delete_raises

    def delete(self):
        if self._delete_raises:
            raise Exception("404 Not Found")
        self._deleted = True


def make_mock_wandb_api(versions):
    """Construct a mock wandb_api with artifact_type(...).collection(...).artifacts() returning versions."""
    mock_api = MagicMock()
    artifact_type_api = MagicMock()
    collection_api = MagicMock()
    # artifacts() yields an iterator over versions
    collection_api.artifacts.return_value = iter(versions)
    artifact_type_api.collection.return_value = collection_api
    mock_api.artifact_type.return_value = artifact_type_api
    return mock_api


def test_cleanup_deletes_old_versions(monkeypatch):
    # Prepare three versions with different creation times
    now = datetime.utcnow()
    v_new = MockVersion('artifact', 3, now)
    v_mid = MockVersion('artifact', 2, now - timedelta(minutes=10))
    v_old = MockVersion('artifact', 1, now - timedelta(minutes=20))

    mock_api = make_mock_wandb_api([v_new, v_mid, v_old])
    monkeypatch.setattr(wandb_logger, 'wandb_api', mock_api)

    # Run cleanup keeping only the latest (1)
    wandb_logger._cleanup_old_checkpoints('artifact', keep_latest=1)

    # After cleanup, middle and old should have been marked deleted
    assert v_mid._deleted is True
    assert v_old._deleted is True
    # Newest should not be deleted
    assert v_new._deleted is False


def test_cleanup_handles_delete_exceptions_and_continues(monkeypatch):
    now = datetime.utcnow()
    v_new = MockVersion('artifact', 3, now)
    v_faulty = MockVersion('artifact', 2, now - timedelta(minutes=10), delete_raises=True)
    v_old = MockVersion('artifact', 1, now - timedelta(minutes=20))

    mock_api = make_mock_wandb_api([v_new, v_faulty, v_old])
    monkeypatch.setattr(wandb_logger, 'wandb_api', mock_api)

    # Should not raise despite one delete raising exception; others should be attempted
    wandb_logger._cleanup_old_checkpoints('artifact', keep_latest=1)

    # Faulty version will not be deleted but should not stop others
    assert v_faulty._deleted is False
    assert v_old._deleted is True


def test_no_cleanup_when_versions_less_or_equal_keep_latest(monkeypatch):
    now = datetime.utcnow()
    v_only = MockVersion('artifact', 1, now)

    mock_api = make_mock_wandb_api([v_only])
    monkeypatch.setattr(wandb_logger, 'wandb_api', mock_api)

    # keep_latest == number of versions -> no deletion
    wandb_logger._cleanup_old_checkpoints('artifact', keep_latest=1)
    assert v_only._deleted is False
