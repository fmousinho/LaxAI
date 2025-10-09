from __future__ import annotations

from datetime import datetime, timezone

import pytest

from services.service_dataprep.src.workflows.manager import DataPrepManager


class FakeStorage:
    bucket_name = "test-bucket"

    def list_blobs(self, *args, **kwargs):  # pragma: no cover - not required here
        return []

    def download_as_string(self, *args, **kwargs):  # pragma: no cover - not required here
        return None


class FakePathManager:
    def get_path(self, key: str, **kwargs) -> str | None:
        if key == "unverified_tracks":
            video_id = kwargs.get("video_id", "video")
            track_id = kwargs.get("track_id", 0)
            return f"process/{video_id}/unverified/{track_id}/"
        if key == "process_root":
            return "process/"
        if key == "dataset_folder":  # pragma: no cover - defensive default
            return "datasets/"
        return None


class FakeStitcherBase:
    def __init__(self):
        self.player_groups: dict[int, set[int]] = {
            10: {1001},
            20: {2001},
        }
        self._in_progress: set[tuple[int, int]] = set()
        self.released_pairs: list[tuple[int, int]] = []
        self.verification_mode = "normal"

    @staticmethod
    def _normalize(group1_id: int, group2_id: int) -> tuple[int, int]:
        g1, g2 = sorted((group1_id, group2_id))
        return g1, g2

    def mark_pair_in_progress(self, group1_id: int, group2_id: int) -> None:
        self._in_progress.add(self._normalize(group1_id, group2_id))

    def release_in_progress_pair(self, group1_id: int, group2_id: int) -> None:
        normalized = self._normalize(group1_id, group2_id)
        if normalized in self._in_progress:
            self._in_progress.remove(normalized)
            self.released_pairs.append(normalized)

    def get_verification_progress(self):  # pragma: no cover - overridden when needed
        return {"total_possible_pairs": 2, "verified_pairs": 0}


class FakeStitcherSuccess(FakeStitcherBase):
    def respond_to_pair(self, group1_id: int, group2_id: int, decision: str, *, mode: str | None = None):
        self.last_response = (group1_id, group2_id, decision, mode)


class FakeStitcherFailure(FakeStitcherBase):
    def respond_to_pair(self, group1_id: int, group2_id: int, decision: str, *, mode: str | None = None):
        raise RuntimeError("forced failure")


class FakeStitcherIssuanceError(FakeStitcherBase):
    def __init__(self):
        super().__init__()
        self.progress_calls = 0

    def get_pair_for_verification(self):
        return {
            "status": "pending_verification",
            "group1_id": 10,
            "group2_id": 20,
            "mode": "normal",
        }

    def get_verification_progress(self):
        self.progress_calls += 1
        raise RuntimeError("progress unavailable")

    def respond_to_pair(self, group1_id: int, group2_id: int, decision: str, *, mode: str | None = None):  # pragma: no cover
        raise AssertionError("respond_to_pair should not be called in issuance error test")


@pytest.fixture
def patched_manager(monkeypatch):
    monkeypatch.setattr(
        "services.service_dataprep.src.workflows.manager.get_storage",
        lambda tenant_id: FakeStorage(),
    )
    monkeypatch.setattr(
        "services.service_dataprep.src.workflows.manager.GCSPaths",
        lambda: FakePathManager(),
    )
    manager = DataPrepManager("tenant")
    manager.current_video_id = "video"
    return manager


def test_record_response_removes_pair(patched_manager):
    manager = patched_manager
    stitcher = FakeStitcherSuccess()
    manager.stitcher = stitcher

    pair = manager._pair_tracker.register_pair(
        group1_id=10,
        group2_id=20,
        mode="normal",
        issued_at=datetime.now(timezone.utc),
        ttl_seconds=600,
    )
    stitcher.mark_pair_in_progress(10, 20)

    response = manager.record_response(pair.pair_id, "same")

    assert response["success"] is True
    assert manager._pair_tracker.active_count == 0
    assert stitcher.released_pairs == [(10, 20)]


def test_record_response_failure_expires_pair(patched_manager):
    manager = patched_manager
    stitcher = FakeStitcherFailure()
    manager.stitcher = stitcher

    pair = manager._pair_tracker.register_pair(
        group1_id=10,
        group2_id=20,
        mode="normal",
        issued_at=datetime.now(timezone.utc),
        ttl_seconds=600,
    )
    stitcher.mark_pair_in_progress(10, 20)

    response = manager.record_response(pair.pair_id, "different")

    assert response["success"] is False
    assert manager._pair_tracker.active_count == 0
    assert stitcher.released_pairs == [(10, 20)]


def test_get_images_rolls_back_on_metadata_error(patched_manager):
    manager = patched_manager
    stitcher = FakeStitcherIssuanceError()
    manager.stitcher = stitcher

    result = manager.get_images_for_verification()

    assert result["status"] == "error"
    assert manager._pair_tracker.active_count == 0
    assert stitcher.released_pairs == []
