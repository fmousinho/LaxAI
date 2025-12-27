"""
PlayerManager - Orchestrates player identity association across track IDs.
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from .player import Player, PlayerState, TrackData
from .config import PlayerConfig

logger = logging.getLogger(__name__)


class PlayerManager:
    """
    Manages player identities across ByteTrack track IDs.
    
    Workflow:
    1. Process each frame's tracks
    2. For new tracks: match to inactive players or create new
    3. For continuing tracks: update player appearance
    4. For ended tracks: deactivate player
    """
    
    def __init__(self, config: Optional[PlayerConfig] = None):
        """
        Initialize PlayerManager.
        
        Args:
            config: Configuration for player management
        """
        self.config = config or PlayerConfig()
        self.players: Dict[int, Player] = {}
        self.track_to_player: Dict[int, int] = {}  # track_id -> player_id
        self.next_player_id: int = 1
        self.current_frame: int = 0
        
        # Track which track IDs we've seen to detect new vs continuing
        self.seen_tracks: set = set()
        self.active_tracks_last_frame: set = set()
    
    def process_frame(
        self,
        frame_id: int,
        tracks: List[TrackData]
    ) -> Dict[int, int]:
        """
        Process all tracks in a frame and return track-to-player mapping.
        
        Args:
            frame_id: Current frame number
            tracks: List of track data for this frame
            
        Returns:
            Dictionary mapping track_id -> player_id
        """
        self.current_frame = frame_id
        active_tracks_this_frame = {t.track_id for t in tracks}
        
        # Detect ended tracks (were active last frame, not in this frame)
        ended_tracks = self.active_tracks_last_frame - active_tracks_this_frame
        for track_id in ended_tracks:
            self._handle_track_end(track_id, frame_id)
        
        # Process current tracks
        for track_data in tracks:
            track_id = track_data.track_id
            
            if track_id not in self.seen_tracks:
                # New track birth
                self._handle_track_birth(track_data)
                self.seen_tracks.add(track_id)
            else:
                # Continuing track
                self._handle_track_continue(track_data)
        
        # Cleanup old inactive players
        if self.config.auto_remove_after > 0:
            self._remove_old_players(frame_id)
        
        self.active_tracks_last_frame = active_tracks_this_frame
        return self.track_to_player.copy()
    
    def _handle_track_birth(self, track_data: TrackData) -> None:
        """
        Handle a new track appearing.
        Try to match to inactive player, or create new.
        """
        matched_player_id = self._match_to_inactive_player(track_data)
        
        if matched_player_id is not None:
            # Reactivate existing player
            player = self.players[matched_player_id]
            player.activate(track_data.track_id, track_data.frame_id)
            self.track_to_player[track_data.track_id] = matched_player_id
            
            # Update appearance
            if track_data.embedding is not None:
                player.update_appearance(track_data.embedding, self.config.embedding_ema_alpha)
            
            if self.config.verbose:
                logger.info(
                    f"Frame {track_data.frame_id}: Matched track {track_data.track_id} "
                    f"to player {matched_player_id} (gap: {track_data.frame_id - player.last_seen_frame})"
                )
        else:
            # Create new player
            player_id = self._create_new_player(track_data)
            
            if self.config.verbose:
                logger.info(
                    f"Frame {track_data.frame_id}: Created new player {player_id} "
                    f"for track {track_data.track_id}"
                )
    
    def _handle_track_continue(self, track_data: TrackData) -> None:
        """Handle a continuing track - update player appearance and frame."""
        player_id = self.track_to_player.get(track_data.track_id)
        if player_id is None:
            logger.warning(f"Track {track_data.track_id} has no associated player")
            return
        
        player = self.players[player_id]
        player.update_frame(track_data.frame_id)
        
        # Update appearance if embedding available
        if track_data.embedding is not None:
            player.update_appearance(track_data.embedding, self.config.embedding_ema_alpha)
    
    def _handle_track_end(self, track_id: int, frame_id: int) -> None:
        """Handle a track ending - deactivate player."""
        player_id = self.track_to_player.get(track_id)
        if player_id is None:
            return
        
        player = self.players[player_id]
        player.deactivate(frame_id)
        
        if self.config.verbose:
            logger.info(
                f"Frame {frame_id}: Track {track_id} ended, "
                f"player {player_id} now inactive"
            )
    
    def _match_to_inactive_player(self, track_data: TrackData) -> Optional[int]:
        """
        Find best matching inactive player for new track.
        
        Args:
            track_data: New track data
            
        Returns:
            player_id if match found, None otherwise
        """
        if track_data.embedding is None:
            return None
        
        candidates: List[Tuple[int, float]] = []
        
        for player in self.players.values():
            # Only match to inactive players
            if player.state != PlayerState.INACTIVE:
                continue
            
            # Check temporal constraints
            gap = track_data.frame_id - player.last_seen_frame
            if gap < self.config.min_inactive_gap or gap > self.config.max_inactive_gap:
                continue
            
            # Check if player has enough embeddings
            if player.embedding_count < self.config.min_embeddings_for_match:
                continue
            
            if player.appearance_mean is None:
                continue
            
            # Compute appearance similarity
            similarity = self._cosine_similarity(
                track_data.embedding,
                player.appearance_mean
            )
            
            # Optional: weight by variance-based confidence
            if self.config.use_variance_weighting and player.appearance_variance is not None:
                confidence = self._compute_confidence(
                    track_data.embedding,
                    player.appearance_mean,
                    player.appearance_variance
                )
                score = similarity * confidence
            else:
                score = similarity
            
            if score >= self.config.similarity_threshold:
                candidates.append((player.player_id, score))
                
                if self.config.verbose:
                    logger.debug(
                        f"Player {player.player_id} candidate: "
                        f"similarity={similarity:.3f}, score={score:.3f}, gap={gap}"
                    )
        
        if not candidates:
            return None
        
        # Return player with highest score
        best_player_id, best_score = max(candidates, key=lambda x: x[1])
        
        if self.config.verbose:
            logger.info(
                f"Best match for track {track_data.track_id}: "
                f"player {best_player_id} (score={best_score:.3f})"
            )
        
        return best_player_id
    
    def _create_new_player(self, track_data: TrackData) -> int:
        """
        Create a new player for an unmatched track.
        
        Args:
            track_data: Track data
            
        Returns:
            New player_id
        """
        player_id = self.next_player_id
        self.next_player_id += 1
        
        player = Player(
            player_id=player_id,
            first_seen_frame=track_data.frame_id,
        )
        
        player.activate(track_data.track_id, track_data.frame_id)
        
        # Initialize appearance
        if track_data.embedding is not None:
            player.update_appearance(track_data.embedding, self.config.embedding_ema_alpha)
        
        self.players[player_id] = player
        self.track_to_player[track_data.track_id] = player_id
        
        return player_id
    
    def _remove_old_players(self, current_frame: int) -> None:
        """Remove inactive players that haven't been seen recently."""
        to_remove = []
        
        for player_id, player in self.players.items():
            if player.state == PlayerState.INACTIVE:
                gap = current_frame - player.last_seen_frame
                if gap > self.config.auto_remove_after:
                    to_remove.append(player_id)
        
        for player_id in to_remove:
            self.players[player_id].state = PlayerState.REMOVED
            if self.config.verbose:
                logger.info(f"Removed player {player_id} (inactive too long)")
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
        
        return float(np.dot(a, b) / (a_norm * b_norm))
    
    @staticmethod
    def _compute_confidence(
        embedding: np.ndarray,
        mean: np.ndarray,
        variance: np.ndarray
    ) -> float:
        """
        Compute confidence based on Mahalanobis-like distance.
        Higher variance = lower confidence.
        """
        epsilon = 1e-6
        inv_var = 1.0 / (variance + epsilon)
        
        diff = embedding - mean
        mahal_dist = np.sqrt(np.sum(diff ** 2 * inv_var))
        
        # Convert distance to confidence (0-1)
        # Lower distance = higher confidence
        confidence = np.exp(-mahal_dist / 10.0)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_player_tracks(self, player_id: int) -> List[int]:
        """Get all track IDs for a player."""
        player = self.players.get(player_id)
        return player.track_ids if player else []
    
    def get_track_player(self, track_id: int) -> Optional[int]:
        """Get player_id for a track_id."""
        return self.track_to_player.get(track_id)
    
    def get_statistics(self) -> dict:
        """Get statistics about player management."""
        total_players = len(self.players)
        active_players = sum(1 for p in self.players.values() if p.state == PlayerState.ACTIVE)
        inactive_players = sum(1 for p in self.players.values() if p.state == PlayerState.INACTIVE)
        removed_players = sum(1 for p in self.players.values() if p.state == PlayerState.REMOVED)
        
        total_tracks = len(self.track_to_player)
        
        return {
            'total_players': total_players,
            'active_players': active_players,
            'inactive_players': inactive_players,
            'removed_players': removed_players,
            'total_tracks': total_tracks,
            'current_frame': self.current_frame,
        }
    
    def export_players(self) -> dict:
        """
        Export all players to dictionary format for JSON serialization.
        
        Returns:
            Dictionary with player data
        """
        return {
            'players': {
                str(player_id): player.to_dict()
                for player_id, player in self.players.items()
                if player.state != PlayerState.REMOVED
            },
            'track_to_player': {
                str(track_id): player_id
                for track_id, player_id in self.track_to_player.items()
            },
            'statistics': self.get_statistics(),
        }
