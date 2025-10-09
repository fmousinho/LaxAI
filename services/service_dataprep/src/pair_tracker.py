"""Utilities for tracking verification pairs issued to clients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional


@dataclass(slots=True)
class ActiveVerificationPair:
    """Represents a verification pair that has been issued to a client."""

    pair_id: str
    group1_id: int
    group2_id: int
    mode: str
    issued_at: datetime
    expires_at: datetime

    def as_tuple(self) -> tuple[int, int]:
        """Return the normalized group tuple for internal tracking."""
        return (min(self.group1_id, self.group2_id), max(self.group1_id, self.group2_id))

    def is_expired(self, reference_time: datetime) -> bool:
        """Whether the pair has expired relative to *reference_time*."""
        return self.expires_at <= reference_time


class VerificationPairTracker:
    """Manages lifecycle bookkeeping for outstanding verification pairs."""

    def __init__(
        self,
        *,
        max_outstanding_pairs: int,
        pair_ttl_seconds: int,
        release_callback: Callable[[int, int], None],
    ) -> None:
        self._pairs: Dict[str, ActiveVerificationPair] = {}
        self._max_outstanding_pairs = max_outstanding_pairs
        self._pair_ttl = pair_ttl_seconds
        self._release_callback = release_callback

    @staticmethod
    def build_pair_id(group1_id: int, group2_id: int) -> str:
        """Generate a deterministic identifier for a pair of group IDs."""
        g1, g2 = sorted((group1_id, group2_id))
        return f"{g1}-{g2}"

    @property
    def max_outstanding_pairs(self) -> int:
        return self._max_outstanding_pairs

    @property
    def active_count(self) -> int:
        return len(self._pairs)

    def outstanding_pair_ids(self) -> List[str]:
        """Return outstanding pair identifiers sorted for stability."""
        return sorted(self._pairs.keys())

    def has_capacity(self) -> bool:
        return self.active_count < self._max_outstanding_pairs

    def cleanup_expired(self, reference_time: Optional[datetime] = None) -> List[str]:
        """Expire all pairs whose TTL has elapsed.

        Returns the list of pair IDs that were expired.
        """
        if not self._pairs:
            return []

        now = reference_time or datetime.utcnow()
        expired_ids: List[str] = []
        for pair_id, pair in list(self._pairs.items()):
            if pair.is_expired(now):
                expired_ids.append(pair_id)
                self._remove_pair(pair_id, pair)

        return expired_ids

    def register_pair(
        self,
        *,
        group1_id: int,
        group2_id: int,
        mode: str,
        issued_at: Optional[datetime] = None,
        ttl_seconds: Optional[int] = None,
    ) -> ActiveVerificationPair:
        """Register a newly issued pair and return its bookkeeping record."""
        issued_time = issued_at or datetime.utcnow()
        ttl = ttl_seconds if ttl_seconds is not None else self._pair_ttl
        expires_at = issued_time + timedelta(seconds=ttl)
        pair_id = self.build_pair_id(group1_id, group2_id)

        pair = ActiveVerificationPair(
            pair_id=pair_id,
            group1_id=group1_id,
            group2_id=group2_id,
            mode=mode,
            issued_at=issued_time,
            expires_at=expires_at,
        )
        self._pairs[pair_id] = pair
        return pair

    def get_pair(self, pair_id: str) -> Optional[ActiveVerificationPair]:
        return self._pairs.get(pair_id)

    def complete_pair(self, pair_id: str, reason: str = "completed") -> Optional[ActiveVerificationPair]:
        """Mark the pair as completed and release its resources."""
        pair = self._pairs.pop(pair_id, None)
        if pair:
            self._release_pair(pair)
        return pair

    def expire_pair(self, pair_id: str, reason: str = "expired") -> Optional[ActiveVerificationPair]:
        """Expire the pair explicitly and release its resources."""
        pair = self._pairs.pop(pair_id, None)
        if pair:
            self._release_pair(pair)
        return pair

    def _remove_pair(self, pair_id: str, pair: ActiveVerificationPair) -> None:
        """Internal helper to drop a pair from bookkeeping and release it."""
        self._pairs.pop(pair_id, None)
        self._release_pair(pair)

    def _release_pair(self, pair: ActiveVerificationPair) -> None:
        self._release_callback(pair.group1_id, pair.group2_id)

    def reset(self) -> None:
        """Clear all outstanding state, releasing associated resources."""
        for pair in list(self._pairs.values()):
            self._release_pair(pair)
        self._pairs.clear()
