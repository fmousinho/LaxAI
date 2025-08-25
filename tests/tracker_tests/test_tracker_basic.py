import numpy as np
import torch

from track.tracker import warp_bbox, TrackData, AffineAwareByteTrack


def test_warp_bbox_identity():
    bbox = np.array([10, 20, 30, 40], dtype=np.float32)
    affine = AffineAwareByteTrack.get_identity_affine_matrix()
    warped = warp_bbox(bbox, affine)
    assert np.allclose(warped, bbox)


def test_warp_bbox_translation():
    bbox = np.array([0, 0, 10, 10], dtype=np.float32)
    # translate by (5, -3)
    affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    warped = warp_bbox(bbox, affine)
    # Note: warp_bbox clips negative coordinates to zero, so expected y_min is 0
    expected = np.array([5.0, 0.0, 15.0, 7.0], dtype=np.float32)
    assert np.allclose(warped, expected)


def test_trackdata_basic_updates():
    crop1 = np.zeros((8, 8, 3), dtype=np.uint8)
    td = TrackData(track_id=7, crop=crop1, class_id=1, confidence=0.6, frame_id=0)

    assert td.track_id == 7
    assert td.num_crops == 1
    assert td.class_id == 1
    assert td.frame_first_seen == 0
    assert td.frame_last_seen == 0

    # Add a new crop with higher confidence, class should update
    crop2 = np.ones((16, 16, 3), dtype=np.uint8)
    td.update_data(crop2, class_id=2, confidence=0.9)
    assert td.num_crops == 2
    assert td.class_id == 2
    assert td.frame_last_seen == 1

    # Update metadata only with lower confidence - should not change class
    td.update_metadata(class_id=3, confidence=0.1)
    assert td.class_id == 2

    # Embedding and team setters
    emb = np.arange(128, dtype=np.float32)
    td.embedding = emb
    td.team = 4
    assert np.array_equal(td.embedding, emb)
    assert td.team == 4


def test_warp_bbox_torch_matches_numpy():
    import torch
    from track.tracker import warp_bbox, warp_bbox_torch

    bboxes = np.array([
        [0, 0, 10, 10],
        [5, 5, 15, 20],
        [10, 10, 12, 12]
    ], dtype=np.float32)

    affine = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float32)

    # numpy baseline
    expected = np.stack([warp_bbox(b, affine) for b in bboxes], axis=0)

    # torch implementation
    t_bboxes = torch.from_numpy(bboxes)
    t_aff = torch.from_numpy(affine)
    warped_t = warp_bbox_torch(t_bboxes, t_aff).cpu().numpy()

    assert np.allclose(expected, warped_t)
