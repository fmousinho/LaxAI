"""
Player data structures for identity management.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class PlayerState(Enum):
    """Player activity state."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    REMOVED = "removed"


@dataclass
class TrackData:
    """
    Per-frame track data from ByteTrack.
    """
    track_id: int
    frame_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    embedding: Optional[np.ndarray] = None


@dataclass
class Player:
    """
    Represents a persistent real-world player across multiple track IDs.
    """
    player_id: int
    
    # Track associations
    track_ids: List[int] = field(default_factory=list)
    """Historical list of all track IDs associated with this player."""
    
    active_track_id: Optional[int] = None
    """Currently active track ID (None if player is inactive)."""
    
    # Temporal info
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    track_segments: List[Tuple[int, int, int]] = field(default_factory=list)
    """List of (track_id, start_frame, end_frame) for each track segment."""
    
    # State
    state: PlayerState = PlayerState.INACTIVE
    
    # Appearance model (Exponential Moving Average)
    appearance_mean: Optional[np.ndarray] = None
    """Mean embedding vector (EMA of all embeddings)."""
    
    appearance_variance: Optional[np.ndarray] = None
    """Variance of embeddings (for confidence estimation)."""
    
    embedding_count: int = 0
    """Number of embeddings incorporated into appearance model."""
    
    # Optional metadata
    team: Optional[str] = None
    confidence: float = 1.0
    
    def update_appearance(self, embedding: np.ndarray, alpha: float = 0.3) -> None:
        """
        Update appearance model with new embedding using exponential moving average.
        
        Args:
            embedding: New embedding vector
            alpha: Weight for new embedding (0-1), higher = more weight to new
        """
        if embedding is None:
            return
            
        if self.appearance_mean is None:
            # First embedding - initialize
            self.appearance_mean = embedding.copy()
            self.appearance_variance = np.zeros_like(embedding)
        else:
            # Update EMA
            delta = embedding - self.appearance_mean
            self.appearance_mean = self.appearance_mean + alpha * delta
            
            # Update variance (Welford's online algorithm adapted for EMA)
            self.appearance_variance = (1 - alpha) * (self.appearance_variance + alpha * delta ** 2)
        
        self.embedding_count += 1
    
    def activate(self, track_id: int, frame_id: int) -> None:
        """Activate player with a new track."""
        self.active_track_id = track_id
        self.state = PlayerState.ACTIVE
        self.last_seen_frame = frame_id
        
        if track_id not in self.track_ids:
            self.track_ids.append(track_id)
            # Start new segment
            self.track_segments.append((track_id, frame_id, frame_id))
    
    def deactivate(self, frame_id: int) -> None:
        """Deactivate player when track ends."""
        if self.active_track_id is not None and self.track_segments:
            # Update last segment's end frame
            last_seg = self.track_segments[-1]
            self.track_segments[-1] = (last_seg[0], last_seg[1], frame_id)
        
        self.active_track_id = None
        self.state = PlayerState.INACTIVE
        self.last_seen_frame = frame_id
    
    def update_frame(self, frame_id: int) -> None:
        """Update last seen frame for active player."""
        self.last_seen_frame = frame_id
        
        # Update active segment's end frame
        if self.track_segments:
            last_seg = self.track_segments[-1]
            self.track_segments[-1] = (last_seg[0], last_seg[1], frame_id)
    
    def to_dict(self) -> dict:
        """Serialize player to dictionary."""
        return {
            'player_id': self.player_id,
            'track_ids': self.track_ids,
            'track_segments': [[int(tid), int(start), int(end)] for tid, start, end in self.track_segments],
            'first_seen_frame': int(self.first_seen_frame),
            'last_seen_frame': int(self.last_seen_frame),
            'state': self.state.value,
            'embedding_count': self.embedding_count,
            'team': self.team,
            'confidence': float(self.confidence),
        }
