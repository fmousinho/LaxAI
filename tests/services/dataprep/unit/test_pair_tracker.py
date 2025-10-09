from datetime import UTC, datetime, timedelta

from services.service_dataprep.src.pair_tracker import VerificationPairTracker


def test_complete_pair_releases_and_removes():
    released = []

    def release_callback(g1: int, g2: int) -> None:
        released.append((g1, g2))

    tracker = VerificationPairTracker(
        max_outstanding_pairs=10,
        pair_ttl_seconds=600,
        release_callback=release_callback,
    )

    now = datetime.now(UTC)
    pair = tracker.register_pair(group1_id=3, group2_id=12, mode="manual", issued_at=now)

    assert tracker.active_count == 1
    assert tracker.outstanding_pair_ids() == [pair.pair_id]

    tracker.complete_pair(pair.pair_id)

    assert tracker.active_count == 0
    assert tracker.outstanding_pair_ids() == []
    assert released == [(3, 12)]


def test_cleanup_expired_pairs():
    released = []

    def release_callback(g1: int, g2: int) -> None:
        released.append((g1, g2))

    tracker = VerificationPairTracker(
        max_outstanding_pairs=10,
        pair_ttl_seconds=600,
        release_callback=release_callback,
    )

    reference_now = datetime.now(UTC)
    issued_at = reference_now - timedelta(seconds=1200)
    pair = tracker.register_pair(group1_id=5, group2_id=9, mode="manual", issued_at=issued_at)

    expired_ids = tracker.cleanup_expired(reference_time=reference_now)

    assert expired_ids == [pair.pair_id]
    assert tracker.active_count == 0
    assert tracker.outstanding_pair_ids() == []
    assert released == [(5, 9)]
