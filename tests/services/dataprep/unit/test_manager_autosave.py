from __future__ import annotations

import signal
import threading
import time

import pytest

from services.service_dataprep.src.workflows.manager import DataPrepManager


class _FakeStorage:
    bucket_name = "test-bucket"

    def list_blobs(self, *args, **kwargs):  # pragma: no cover - unused in these tests
        return []

    def download_as_string(self, *args, **kwargs):  # pragma: no cover - unused in these tests
        return None

    def upload_from_file(self, *args, **kwargs):  # pragma: no cover - unused in these tests
        return True


class _FakePaths:
    def get_path(self, key: str, **kwargs) -> str | None:
        if key == "saved_graph":
            return "process/video/saved_graph.gml"
        if key == "process_root":  # pragma: no cover - defensive default
            return "process/"
        return "unused"


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv("DATAPREP_AUTOSAVE_INTERVAL_SECONDS", "1")
    monkeypatch.setattr(
        "services.service_dataprep.src.workflows.manager.get_storage",
        lambda tenant_id: _FakeStorage(),
    )
    monkeypatch.setattr(
        "services.service_dataprep.src.workflows.manager.GCSPaths",
        lambda: _FakePaths(),
    )

    mgr = DataPrepManager("tenant")
    mgr.stitcher = None
    mgr.current_video_id = "video"

    try:
        yield mgr
    finally:
        mgr._stop_autosave_loop()


def test_autosave_loop_triggers_periodic_saves(manager, monkeypatch):
    manager._autosave_interval_seconds = 0.1

    save_event = threading.Event()
    reasons: list[str] = []

    def fake_save(reason: str) -> bool:
        reasons.append(reason)
        save_event.set()
        return True

    monkeypatch.setattr(manager, "_save_graph_internal", fake_save)

    manager._start_autosave_loop()
    try:
        assert save_event.wait(1.0), "autosave loop did not invoke save within expected time"
    finally:
        manager._stop_autosave_loop()

    assert any(reason == "autosave" for reason in reasons)


def test_stop_autosave_loop_cleans_up_thread(manager, monkeypatch):
    manager._autosave_interval_seconds = 0.1

    call_count = 0
    first_call = threading.Event()

    def fake_save(reason: str) -> bool:
        nonlocal call_count
        call_count += 1
        first_call.set()
        return True

    monkeypatch.setattr(manager, "_save_graph_internal", fake_save)

    manager._start_autosave_loop()
    assert first_call.wait(1.0), "autosave loop never invoked save"

    manager._stop_autosave_loop()

    time.sleep(0.2)

    assert call_count >= 1
    assert manager._autosave_thread is None
    assert manager._autosave_stop_event is None


def test_schedule_shutdown_save_invokes_background_persistence(manager, monkeypatch):
    stop_called = threading.Event()
    save_called = threading.Event()
    reasons: list[str] = []

    def fake_stop():
        stop_called.set()

    def fake_save(reason: str) -> bool:
        reasons.append(reason)
        save_called.set()
        return True

    monkeypatch.setattr(manager, "_stop_autosave_loop", fake_stop)
    monkeypatch.setattr(manager, "_save_graph_internal", fake_save)

    manager._schedule_shutdown_save(signal.SIGTERM)

    assert stop_called.wait(1.0), "shutdown save did not attempt to stop autosave loop"
    assert save_called.wait(1.0), "shutdown save did not persist graph"

    deadline = time.time() + 1.0
    while manager._shutdown_in_progress and time.time() < deadline:
        time.sleep(0.05)

    assert not manager._shutdown_in_progress, "shutdown worker did not complete"
    assert reasons == ["shutdown"]
