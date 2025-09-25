import json
from typing import List, Dict, Any, cast

import numpy as np
import supervision as sv
import pytest

from shared_libs.common.detection_utils import save_all_detections, load_all_detections_summary


class DummyStorageClient:
    def __init__(self):
        self.uploads = {}

    def upload_from_bytes(self, destination_blob_name: str, data: bytes, content_type=None):  # noqa: D401
        # Simulate GCS upload. Store raw bytes for later inspection.
        self.uploads[destination_blob_name] = data
        return True


def _make_detections(n: int, frame_idx: int) -> sv.Detections:
    if n == 0:
        return sv.Detections.empty()
    xyxy = np.random.randint(0, 100, size=(n, 4)).astype(float)
    class_id = np.random.randint(0, 3, size=(n,))
    confidence = np.random.random(size=(n,)).astype(float)
    tracker_id = np.arange(n)
    # Use untyped list values and cast to satisfy strict typing expectations
    raw_data: Dict[str, Any] = {"frame_index": [int(frame_idx)] * n, "custom_field": list(range(n))}
    d = sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        tracker_id=tracker_id,
        data=cast(dict, raw_data),
    )
    # also set metadata frame id for redundancy
    d.metadata.update({"frame_id": frame_idx})
    return d


def test_save_all_detections_empty_list():
    storage = DummyStorageClient()
    ok = save_all_detections(storage, "detections.json", [])
    assert ok is True
    assert "detections.json" in storage.uploads
    payload = json.loads(storage.uploads["detections.json"].decode("utf-8"))
    assert payload["total_frames"] == 0
    assert payload["total_detections"] == 0
    assert payload["xyxy"] == []


def test_save_all_detections_non_empty_and_load_round_trip():
    storage = DummyStorageClient()
    det_list: List[sv.Detections] = [
        _make_detections(3, 0),
        _make_detections(2, 1),
    ]
    ok = save_all_detections(storage, "video_root/detections.json", det_list)
    assert ok is True
    raw = storage.uploads["video_root/detections.json"].decode("utf-8")
    parsed = json.loads(raw)
    assert parsed["total_frames"] == len(det_list)
    assert parsed["total_detections"] == sum(len(d) for d in det_list)
    # Ensure frame_index length matches total rows
    assert len(parsed["frame_index"]) == parsed["total_detections"]

    merged = load_all_detections_summary(parsed)
    assert isinstance(merged, sv.Detections)
    assert len(merged) == parsed["total_detections"]
    # Confirm frame_index recovered
    assert "frame_index" in merged.data
    assert len(merged.data["frame_index"]) == len(merged)


def test_load_all_detections_summary_handles_missing_fields():
    minimal = {
        "xyxy": [],
        "confidence": [],
        "class_id": [],
        "tracker_id": [],
        "frame_index": [],
        "data": {},
        "total_frames": 0,
        "total_detections": 0,
    }
    det = load_all_detections_summary(minimal)
    assert isinstance(det, sv.Detections)
    assert len(det) == 0


@pytest.mark.parametrize("n1,n2", [(1, 1), (2, 3)])
def test_frame_index_fallback_metadata(n1, n2):
    storage = DummyStorageClient()
    d1 = _make_detections(n1, 5)
    d2 = _make_detections(n2, 6)
    # Remove explicit frame_index list to force metadata fallback for second
    del d2.data["frame_index"]
    det_list = [d1, d2]
    ok = save_all_detections(storage, "f/detections.json", det_list)
    assert ok
    parsed = json.loads(storage.uploads["f/detections.json"].decode("utf-8"))
    assert parsed["total_detections"] == n1 + n2
    assert len(parsed["frame_index"]) == n1 + n2
    # Ensure both frame ids appear
    assert 5 in parsed["frame_index"] and 6 in parsed["frame_index"]
