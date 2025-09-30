import json
from typing import List, Dict, Any, cast

import numpy as np
import supervision as sv
import pytest

from shared_libs.common.detection_utils import detections_to_json, json_to_detections


class DummyStorageClient:
    def __init__(self):
        self.uploads = {}

    def upload_from_bytes(self, destination_blob_name: str, data, content_type=None):  # noqa: D401
        # Simulate GCS upload. Store raw bytes for later inspection.
        # Handle both bytes and dict data (like real GoogleStorageClient)
        if destination_blob_name.endswith(".json") and isinstance(data, dict):
            import json
            json_bytes = json.dumps(data).encode("utf-8")
            self.uploads[destination_blob_name] = json_bytes
        else:
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


def test_detections_to_json_empty_list():
    detections_list = []
    json_list = detections_to_json(detections_list)
    assert json_list == []
    
    # Test round trip
    restored_list = json_to_detections(json_list)
    assert len(restored_list) == 0


def test_detections_to_json_round_trip():
    det_list: List[sv.Detections] = [
        _make_detections(3, 0),
        _make_detections(2, 1),
    ]
    
    # Convert to JSON list
    json_list = detections_to_json(det_list)
    assert len(json_list) == 2
    assert len(json_list[0]["xyxy"]) == 3  # First detection has 3 objects
    assert len(json_list[1]["xyxy"]) == 2  # Second detection has 2 objects
    
    # Test round trip conversion
    restored_list = json_to_detections(json_list)
    assert len(restored_list) == 2
    assert len(restored_list[0]) == 3
    assert len(restored_list[1]) == 2
    
    # Verify data integrity
    assert np.allclose(restored_list[0].xyxy, det_list[0].xyxy)
    assert np.allclose(restored_list[1].xyxy, det_list[1].xyxy)


def test_json_to_detections_handles_empty_json():
    json_list = []
    det_list = json_to_detections(json_list)
    assert len(det_list) == 0


@pytest.mark.parametrize("n1,n2", [(1, 1), (2, 3)])
def test_frame_index_preservation(n1, n2):
    d1 = _make_detections(n1, 5)
    d2 = _make_detections(n2, 6)
    det_list = [d1, d2]
    
    # Convert to JSON and back
    json_list = detections_to_json(det_list)
    restored_list = json_to_detections(json_list)
    
    # Verify frame indices are preserved
    assert len(restored_list[0].data["frame_index"]) == n1
    assert all(idx == 5 for idx in restored_list[0].data["frame_index"])
    assert len(restored_list[1].data["frame_index"]) == n2
    assert all(idx == 6 for idx in restored_list[1].data["frame_index"])
